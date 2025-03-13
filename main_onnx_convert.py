import os
import sys
import json
import torch
import onnx
from torch import nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models import get_model

resnet_backbones = ["resnet50", "resnet101", "resnet50x4", "resnet50x16", "resnet50x64"]
vit_backbones = ["vit_b_16", "vit_b_32", "vit_l_14", "vit_l_14_336px"]

class CLIP_EBC_Wrapper(nn.Module):
    """
    ONNX export 용 CLIP_EBC 래퍼 클래스
    - 항상 inference 모드로 동작하도록 보장
    - 텍스트 인코더의 출력을 미리 계산하여 모델에 포함
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()  # 항상 평가 모드로 설정
        
        # 백본 타입 저장
        self.backbone = self.model.backbone
        
        # 텍스트 특성을 미리 계산하여 모델의 매개변수로 저장
        if hasattr(self.model, 'text_features') and self.model.text_features is not None:
            # 이미 계산된 텍스트 특성이 있는 경우
            self.text_features = nn.Parameter(
                self.model.text_features.clone(),
                requires_grad=False
            )
        else:
            # 텍스트 특성을 계산해야 하는 경우
            with torch.no_grad():
                text_features = self.model.text_encoder(self.model.text_prompts)
                self.text_features = nn.Parameter(
                    text_features.clone(),
                    requires_grad=False
                )
        
        # 앵커 포인트를 모델 매개변수로 저장
        self.anchor_points = nn.Parameter(
            self.model.anchor_points.clone(),
            requires_grad=False
        )
        
        # logit_scale도 저장
        self.logit_scale = nn.Parameter(
            self.model.logit_scale.clone(),
            requires_grad=False
        )

    def forward(self, x):
        device = x.device
        
        # 이미지 인코더 및 디코더 부분
        if self.backbone in resnet_backbones:
            x = self.model.image_encoder(x)
        else:
            x = self._forward_vpt(x)
            
        if self.model.reduction != self.model.encoder_reduction:
            x = torch.nn.functional.interpolate(
                x, 
                scale_factor=self.model.encoder_reduction / self.model.reduction, 
                mode="bilinear"
            )
        
        x = self.model.image_decoder(x)
        x = self.model.projection(x)
        
        # 이미지 특성 정규화 및 로짓 계산
        image_features = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        text_features = self.text_features.to(device)  # (N, C)
        
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        
        # 코사인 유사도 계산
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # (B, H, W, N)
        logits = logits.permute(0, 3, 1, 2)  # (B, N, H, W)
        
        # 확률 및 기대값 계산
        probs = logits.softmax(dim=1)
        exp = (probs * self.anchor_points.to(device)).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        return exp
        
    def _forward_vpt(self, x):
        """
        ViT 모델용 VPT 처리 함수
        원본 모델에서 복사
        """
        device = x.device
        batch_size, _, height, width = x.shape
        num_h_patches = height // self.model.image_encoder.patch_size[0]
        num_w_patches = width // self.model.image_encoder.patch_size[1]

        image_features = self.model.image_encoder.conv1(x)
        image_features = image_features.reshape(batch_size, image_features.shape[1], -1)
        image_features = image_features.permute(0, 2, 1)  # (B, num_patches, C)
        image_features = torch.cat([
            self.model.image_encoder.class_embedding + torch.zeros(
                batch_size, 1, image_features.shape[-1], 
                dtype=image_features.dtype, device=device
            ),
            image_features,
        ], dim=1)  # (B, num_patches + 1, C)

        pos_embedding = self.model.image_encoder._interpolate_pos_embed(num_h_patches, num_w_patches)
        image_features = image_features + pos_embedding
        image_features = self.model.image_encoder.ln_pre(image_features)
        image_features = image_features.permute(1, 0, 2)  # (num_patches + 1, B, C)

        # VPT 프롬프트 준비
        vpt = self._prepare_vpt(0, batch_size, device)
        
        # 트랜스포머 레이어 통과
        for idx in range(self.model.image_encoder_depth):
            # 어셈블
            image_features = torch.cat([
                image_features[:1, :, :],  # CLS 토큰
                vpt,
                image_features[1:, :, :],
            ], dim=0)

            # 트랜스포머
            image_features = self.model.image_encoder.transformer.resblocks[idx](image_features)

            # 디스어셈블
            if idx < self.model.image_encoder_depth - 1:
                if self.model.deep_vpt:
                    vpt = self._prepare_vpt(idx + 1, batch_size, device)
                else:
                    vpt = image_features[1: (self.model.num_vpt + 1), :, :]

            image_features = torch.cat([
                image_features[:1, :, :],  # CLS 토큰
                image_features[(self.model.num_vpt + 1):, :, :],
            ], dim=0)
            
        image_features = image_features.permute(1, 0, 2)  # (B, num_patches + 1, C)
        image_features = self.model.image_encoder.ln_post(image_features)
        image_features = image_features[:, 1:, :].permute(0, 2, 1)  # (B, C, num_patches)
        image_features = image_features.reshape(batch_size, -1, num_h_patches, num_w_patches)
        return image_features

    def _prepare_vpt(self, layer, batch_size, device):
        """
        VPT 준비 함수
        원본 모델에서 복사
        """
        if not self.model.deep_vpt:
            assert layer == 0, f"Expected layer to be 0 when using Shallow VPT, got {layer}"

        vpt = getattr(self.model, f"vpt_{layer}").to(device)
        vpt = vpt.unsqueeze(0).expand(batch_size, -1, -1)
        vpt = getattr(self.model, f"vpt_drop_{layer}")(vpt)
        vpt = vpt.permute(1, 0, 2)  # (num_vpt, batch_size, hidden_dim)
        return vpt


def export_clip_ebc_to_onnx():
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 모델 파라미터 설정
    truncation = 4
    reduction = 8
    granularity = "fine"
    anchor_points_type = "average"
    model_name = "clip_vit_b_16"
    input_size = 224
    prompt_type = "word"
    dataset_name = "qnrf"
    num_vpt = 32
    vpt_drop = 0.0
    deep_vpt = True
    
    # 설정 파일 로드
    config_dir = "configs"
    config_path = os.path.join(config_dir, f"reduction_{reduction}.json")
    with open(config_path, "r") as f:
        config = json.load(f)[str(truncation)][dataset_name]
    
    bins = config["bins"][granularity]
    bins = [(float(b[0]), float(b[1])) for b in bins]
    
    if anchor_points_type == "average":
        anchor_points = config["anchor_points"][granularity]["average"]
    else:
        anchor_points = config["anchor_points"][granularity]["middle"]
    anchor_points = [float(p) for p in anchor_points]
    
    # 모델 초기화
    print("모델 초기화 중...")
    model = get_model(
        backbone=model_name,
        input_size=input_size,
        reduction=reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=prompt_type,
        num_vpt=num_vpt,
        vpt_drop=vpt_drop,
        deep_vpt=deep_vpt
    )
    
    # 체크포인트 로드
    print("체크포인트 로드 중...")
    ckpt_path = "assets/CLIP_EBC_nwpu_rmse.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    
    # 모델 구조 디버깅
    print(f"모델 타입: {type(model).__name__}")
    print(f"백본 이름: {model.backbone}")
    
    # ONNX 래퍼 모델 생성
    print("ONNX 래퍼 모델 생성 중...")
    wrapped_model = CLIP_EBC_Wrapper(model)
    wrapped_model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
    # ONNX 변환 전 PyTorch 출력 확인
    print("PyTorch 모델 추론 테스트 중...")
    with torch.no_grad():
        torch_output = wrapped_model(dummy_input)
    
    print(f"PyTorch 모델 출력 형태: {torch_output.shape}")
    
    # ONNX 변환
    print("ONNX 모델로 변환 중...")
    output_path = "clip_ebc_model.onnx"
    
    # ONNX 내보내기 옵션
    torch.onnx.export(
        wrapped_model,               # 래핑된 모델
        dummy_input,                 # 더미 입력
        output_path,                 # 출력 파일 경로
        export_params=True,          # 모델 파라미터 내보내기
        opset_version=17,            # ONNX 버전
        do_constant_folding=True,    # 상수 폴딩 수행
        input_names=['input'],       # 입력 이름
        output_names=['output'],     # 출력 이름
        dynamic_axes={
            'input': {0: 'batch_size'},   # 배치 크기 동적 처리
            'output': {0: 'batch_size'}   # 출력 배치 크기 동적 처리
        }
    )
    
    # ONNX 모델 검증
    print("ONNX 모델 검증 중...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX 모델이 성공적으로 생성되고 검증되었습니다! 저장 경로: {output_path}")
    # ONNX 모델 추론 검증 (ONNX Runtime 필요)
    try:
        import onnxruntime as ort
        print("ONNX Runtime을 사용하여 출력 검증 중...")
        
        # 세션 생성
        ort_session = ort.InferenceSession(output_path)
        
        # 입력 준비
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        
        # 추론 실행
        ort_outs = ort_session.run(None, ort_inputs)
        
        # PyTorch 출력과 비교
        pytorch_output = torch_output.cpu().numpy()
        onnx_output = ort_outs[0]
        
        # 출력 비교
        max_diff = abs(pytorch_output - onnx_output).max()
        print(f"PyTorch와 ONNX 출력 최대 절대 차이: {max_diff}")
        
        if max_diff < 1e-5:
            print("검증 성공: PyTorch와 ONNX 출력이 일치합니다!")
        else:
            print("주의: PyTorch와 ONNX 출력에 차이가 있습니다.")
            
    except ImportError:
        print("ONNX Runtime이 설치되지 않았습니다. 출력 검증을 건너뜁니다.")
        print("검증하려면 'pip install onnxruntime' 명령으로 설치하세요.")
    
if __name__ == "__main__":
    export_clip_ebc_to_onnx()