import onnx
import tensorrt as trt
import os

onnx_model_path = "/home/jungseoik/data/PR/CLIP_EBC_ONNX/assets/CLIP_EBC_nwpu_rmse_onnx.onnx"
engine_path = "/home/jungseoik/data/PR/CLIP_EBC_ONNX/assets/CLIP_EBC_nwpu_rmse_tensorrt.trt"

# ONNX 모델 로드 및 체크
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX 모델이 정상적으로 로드되었습니다.")

# TensorRT 로거 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    # TensorRT 빌더 생성
    builder = trt.Builder(TRT_LOGGER)
    
    # 명시적 배치 크기 설정으로 네트워크 생성
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # ONNX 파서 생성
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # ONNX 모델 파싱
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: ONNX 모델 파싱 실패")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 빌더 설정 생성
    config = builder.create_builder_config()
    
    # 작업 공간 크기 설정 (2GB로 증가)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    
    # FP16 정밀도 사용 (성능 향상)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 모드 활성화")
    
    # 최적화 프로파일 생성
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    
    # 동적 배치 크기 설정
    min_shape = (1, 3, 224, 224)
    opt_shape = (4, 3, 224, 224)
    max_shape = (16, 3, 224, 224)
    
    # 최소/최적/최대 형태 설정
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # 디버깅 정보 출력
    print(f"Input name: {input_name}")
    print(f"Input shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
    
    # 네트워크 입출력 정보 출력
    print("\n=== 네트워크 정보 ===")
    print(f"입력 개수: {network.num_inputs}")
    print(f"출력 개수: {network.num_outputs}")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"입력 {i}: 이름={input_tensor.name}, 형태={input_tensor.shape}, 데이터 타입={input_tensor.dtype}")
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"출력 {i}: 이름={output_tensor.name}, 형태={output_tensor.shape}, 데이터 타입={output_tensor.dtype}")
    
    # 엔진 빌드
    print("\n엔진 빌드 중...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: TensorRT 엔진 생성 실패")
        return None
    
    # 엔진 역직렬화
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    return engine, serialized_engine

# 엔진 빌드
engine, serialized_engine = build_engine(onnx_model_path)

if engine:
    print("TensorRT 엔진 변환 성공!")
    
    # 엔진 저장
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT 엔진 저장 완료: {engine_path}")
    
    # 엔진 정보 출력
    print("\n=== 생성된 엔진 정보 ===")
    print(f"엔진에서 I/O 텐서 개수: {engine.num_io_tensors}")
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        
        mode_str = "입력" if mode == trt.TensorIOMode.INPUT else "출력"
        print(f"텐서 {i}: 이름={name}, 모드={mode_str}, 형태={shape}, 데이터 타입={dtype}")
else:
    print("엔진 생성 실패")