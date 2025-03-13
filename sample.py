# 기존 코드
from custom.clip_ebc import ClipEBC

model = ClipEBC()
count = model.predict('assets/289.jpg')
print("-"*100)
print(count)

# ONNX 버전
from custom.clip_ebc_onnx import ClipEBCOnnx
model = ClipEBCOnnx(onnx_model_path='clip_ebc_model.onnx')
count = model.predict('assets/289.jpg')
print("-"*100)
print(count)