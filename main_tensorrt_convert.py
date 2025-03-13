import tensorrt as trt
import os

# TensorRT 로거 생성
logger = trt.Logger(trt.Logger.WARNING)

# 입력 및 출력 파일 경로
onnx_model_path = "assets/CLIP_EBC_nwpu_rmse_onnx.onnx"
engine_path = "assets/CLIP_EBC_nwpu_rmse_tensorrt.engine"

# TensorRT 버전 확인
print(f"TensorRT 버전: {trt.__version__}")

# TensorRT 빌더 생성
builder = trt.Builder(logger)

# 네트워크 정의 생성 (EXPLICIT_BATCH 플래그 사용)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# ONNX 파서 생성
parser = trt.OnnxParser(network, logger)

# ONNX 모델 파일 파싱
print(f"ONNX 모델 '{onnx_model_path}' 파싱 중...")
with open(onnx_model_path, 'rb') as model:
    if not parser.parse(model.read()):
        print("ONNX 모델 파싱 실패!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
print("ONNX 모델 파싱 완료")

# 빌더 구성 설정
config = builder.create_builder_config()

# 최신 TensorRT 버전용 메모리 설정 (1GB)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# FP16 정밀도 설정 (하드웨어가 지원하는 경우)
if builder.platform_has_fast_fp16:
    print("FP16 모드 활성화")
    config.set_flag(trt.BuilderFlag.FP16)
else:
    print("FP16을 지원하지 않음, FP32 모드 사용")

# 동적 입력을 위한 최적화 프로필 설정
print("동적 입력을 위한 최적화 프로필 설정 중...")
profile = builder.create_optimization_profile()

# 모델 입력 정보 출력
print("모델 입력 정보:")
for i in range(network.num_inputs):
    input_tensor = network.get_input(i)
    input_name = input_tensor.name
    input_shape = input_tensor.shape
    input_dtype = input_tensor.dtype
    print(f"  입력 {i}: 이름={input_name}, 형태={input_shape}, 타입={input_dtype}")

# 첫 번째 입력에 대한 최적화 프로필 설정 (예시)
input_name = network.get_input(0).name
input_shape = network.get_input(0).shape
min_batch = 1
opt_batch = 1
max_batch = 32

# 동적 배치 크기에 대한 최적화 프로필 설정
# 첫 번째 차원(배치 크기)만 동적으로 설정하고 나머지 차원은 고정
print(f"입력 '{input_name}'에 대한 최적화 프로필 설정")
min_shape = (min_batch,) + tuple(input_shape[1:])
opt_shape = (opt_batch,) + tuple(input_shape[1:])
max_shape = (max_batch,) + tuple(input_shape[1:])

print(f"  최소 형태: {min_shape}")
print(f"  최적 형태: {opt_shape}")
print(f"  최대 형태: {max_shape}")

profile.set_shape(input_name, min_shape, opt_shape, max_shape)
config.add_optimization_profile(profile)

# 최신 TensorRT API 사용
print("TensorRT 엔진 빌드 중...")
try:
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("엔진 생성 실패!")
        exit()

    # 엔진 파일로 저장
    print(f"TensorRT 엔진을 '{engine_path}'에 저장 중...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print("변환 완료!")
    print(f"TensorRT 엔진이 {engine_path}에 저장되었습니다.")
except Exception as e:
    print(f"엔진 빌드 중 오류 발생: {e}")