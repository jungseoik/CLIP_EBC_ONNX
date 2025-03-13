import os
import sys
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from main_onnx_convert import CLIP_EBC_Wrapper

# 프로젝트 루트 디렉토리 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def compare_outputs(onnx_path, batch_size=1, input_size=224):
    """
    PyTorch 모델과 ONNX 모델의 출력을 비교하고 시각화합니다.
    
    Args:
        onnx_path: ONNX 모델 파일 경로
        batch_size: 입력 배치 크기
        input_size: 입력 이미지 크기
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 이미 변환된 ONNX 모델을 불러오기 위한 Wrapper 모델 필요
    # 1. 먼저 같은 입력으로 PyTorch 모델과 ONNX 모델 출력 생성
    
    # 동일한 랜덤 입력 생성 (재현 가능하도록 시드 고정)
    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
    
    # PyTorch 모델을 로드하고 실행
    # 아래 코드는 이미 래핑된 모델이 있다고 가정하고 간소화된 형태입니다
    from models import get_model
    
    # 모델 파라미터 설정 (필요에 따라 조정)
    truncation = 4
    reduction = 8
    granularity = "fine"
    anchor_points_type = "average"
    model_name = "clip_vit_b_16" 
    prompt_type = "word"
    dataset_name = "qnrf"
    num_vpt = 32
    vpt_drop = 0.0
    deep_vpt = True
    
    # 설정 파일 로드
    config_dir = "configs"
    config_path = os.path.join(config_dir, f"reduction_{reduction}.json")
    with open(config_path, "r") as f:
        import json
        config = json.load(f)[str(truncation)][dataset_name]
    
    bins = config["bins"][granularity]
    bins = [(float(b[0]), float(b[1])) for b in bins]
    
    if anchor_points_type == "average":
        anchor_points = config["anchor_points"][granularity]["average"]
    else:
        anchor_points = config["anchor_points"][granularity]["middle"]
    anchor_points = [float(p) for p in anchor_points]
    
    # 모델 초기화 및 설정
    print("PyTorch 모델 초기화 중...")
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
    ckpt_path = "assets/CLIP_EBC_nwpu_rmse.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    
    # ONNX 래퍼 모델 생성
    resnet_backbones = ["resnet50", "resnet101", "resnet50x4", "resnet50x16", "resnet50x64"]
    wrapped_model = CLIP_EBC_Wrapper(model)
    wrapped_model.eval()
    
    # PyTorch 출력 계산
    print("PyTorch 모델 출력 계산 중...")
    with torch.no_grad():
        pytorch_output = wrapped_model(dummy_input).cpu().numpy()
    
    # ONNX 세션 생성 및 출력 계산
    print("ONNX 모델 출력 계산 중...")
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # 출력 형태 확인
    print(f"PyTorch 출력 형태: {pytorch_output.shape}")
    print(f"ONNX 출력 형태: {onnx_output.shape}")
    
    # 두 출력의 절대 차이 계산
    abs_diff = np.abs(pytorch_output - onnx_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    print(f"최대 절대 차이: {max_diff}")
    print(f"평균 절대 차이: {mean_diff}")
    
    # 출력 시각화
    # 1. 샘플 이미지의 경우 채널 1개만 시각화
    sample_idx = 0  # 첫 번째 배치 이미지 선택
    
    # 가장 많은 차이를 보이는 위치 찾기
    max_diff_pos = np.unravel_index(np.argmax(abs_diff[sample_idx, 0]), abs_diff[sample_idx, 0].shape)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # PyTorch 출력 시각화
    im0 = axes[0, 0].imshow(pytorch_output[sample_idx, 0], cmap='viridis')
    axes[0, 0].set_title('PyTorch 출력')
    axes[0, 0].set_xlabel('너비')
    axes[0, 0].set_ylabel('높이')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    
    # ONNX 출력 시각화
    im1 = axes[0, 1].imshow(onnx_output[sample_idx, 0], cmap='viridis')
    axes[0, 1].set_title('ONNX 출력')
    axes[0, 1].set_xlabel('너비')
    axes[0, 1].set_ylabel('높이')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    
    # 절대 차이 시각화
    im2 = axes[0, 2].imshow(abs_diff[sample_idx, 0], cmap='hot')
    axes[0, 2].set_title(f'절대 차이 (최대: {max_diff:.8f})')
    axes[0, 2].set_xlabel('너비')
    axes[0, 2].set_ylabel('높이')
    axes[0, 2].plot(max_diff_pos[1], max_diff_pos[0], 'rx', markersize=10)  # 최대 차이 위치 표시
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    # 로그 스케일로 절대 차이 시각화
    im3 = axes[1, 0].imshow(abs_diff[sample_idx, 0], cmap='hot', norm=LogNorm())
    axes[1, 0].set_title('절대 차이 (로그 스케일)')
    axes[1, 0].set_xlabel('너비')
    axes[1, 0].set_ylabel('높이')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')
    
    # 히스토그램 시각화
    axes[1, 1].hist(abs_diff[sample_idx, 0].flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('절대 차이 히스토그램')
    axes[1, 1].set_xlabel('절대 차이')
    axes[1, 1].set_ylabel('빈도')
    axes[1, 1].set_yscale('log')
    
    # 상대 오차 히스토그램
    # 0으로 나누기 오류 방지
    mask = pytorch_output[sample_idx, 0] != 0
    rel_diff = np.zeros_like(pytorch_output[sample_idx, 0])
    rel_diff[mask] = abs_diff[sample_idx, 0][mask] / np.abs(pytorch_output[sample_idx, 0][mask]) * 100
    
    axes[1, 2].hist(rel_diff[mask].flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('상대 오차 히스토그램 (%)')
    axes[1, 2].set_xlabel('상대 오차 (%)')
    axes[1, 2].set_ylabel('빈도')
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('pytorch_vs_onnx_comparison.png', dpi=300)
    print(f"시각화 이미지가 저장되었습니다: pytorch_vs_onnx_comparison.png")
    
    # 추가적인 통계 계산
    print("\n추가 통계:")
    percentiles = [50, 90, 95, 99, 99.9]
    for p in percentiles:
        print(f"{p}번째 백분위수 차이: {np.percentile(abs_diff, p):.8f}")
    
    return pytorch_output, onnx_output, abs_diff

if __name__ == "__main__":
    # ONNX 모델 경로 지정
    onnx_path = "clip_ebc_model.onnx"
    
    # 출력 비교 실행
    pytorch_output, onnx_output, abs_diff = compare_outputs(onnx_path)