# 기존 코드
from custom.clip_ebc import ClipEBC

model = ClipEBC()
count = model.predict('assets/289.jpg')
print("-"*100)
print(count)

# ONNX 버전
from custom.clip_ebc_onnx import ClipEBCOnnx
model = ClipEBCOnnx()
count = model.predict('assets/289.jpg')
print("-"*100)
print(count)

from custom.clip_ebc_tensorrt import ClipEBCTensorRT
model = ClipEBCTensorRT(
    engine_path="assets/CLIP_EBC_nwpu_rmse_tensorrt.engine"
)
count = model.predict('assets/289.jpg')
print(f"예측된 사람 수: {count:.1f}")
print("-"*100)
print(count)