import os
import sys
import torch
import onnx
from models import get_model  # 모델 불러오기

# 프로젝트 루트 디렉토리 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 파라미터 설정 (ClipEBC 클래스와 동일하게)
truncation = 4
reduction = 8
granularity = "fine"
anchor_points_type = "average"
model_name = "clip_vit_b_16"
input_size = 224
window_size = 224
stride = 224
prompt_type = "word"
dataset_name = "qnrf"
num_vpt = 32
vpt_drop = 0.0
deep_vpt = True
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

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

# 모델 초기화
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

# 더미 입력 생성 (동적 배치 크기)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# ONNX 변환
torch.onnx.export(
    model,                      # 모델
    dummy_input,                # 더미 입력
    "clip_ebc_model.onnx",      # 출력 파일 이름
    export_params=True,         # 모델 파라미터 저장
    opset_version=17,           # ONNX 버전 (설치된 onnx 1.17.0에 맞게)
    do_constant_folding=True,   # 상수 폴딩 최적화
    input_names=['input'],      # 입력 이름
    output_names=['output'],    # 출력 이름
    dynamic_axes={
        'input': {0: 'batch_size'},    # 배치 크기만 동적
        'output': {0: 'batch_size'}    # 출력도 배치 크기만 동적
    }
)

# 모델 검증
onnx_model = onnx.load("clip_ebc_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 모델이 성공적으로 생성되고 검증되었습니다!")

