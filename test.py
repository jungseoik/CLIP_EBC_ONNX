import tensorrt as trt
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import time
import os
import gc  # 가비지 컬렉션 추가
import torch
torch.cuda.empty_cache()

# TensorRT 엔진 파일 경로
engine_path = "/home/jungseoik/data/PR/CLIP_EBC_ONNX/assets/CLIP_EBC_nwpu_rmse_tensorrt.trt"

# TensorRT 로거 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    """TensorRT 엔진 파일 로드"""
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    return engine
def allocate_buffers(engine):
    """TensorRT 엔진 실행을 위한 메모리 버퍼 할당"""
    bindings = []
    input_buffers = {}
    output_buffers = {}
    
    gc.collect()
    cuda.Context.synchronize()
    
    try:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            
            # -1을 실제 배치 크기로 변경
            shape = tuple([s if s > 0 else 1 for s in shape])
            
            print(f"텐서 {i}: 이름={name}, 모드={'입력' if mode == trt.TensorIOMode.INPUT else '출력'}, 형태={shape}")

            if dtype == trt.DataType.FLOAT:
                np_dtype = np.float32
            elif dtype == trt.DataType.HALF:
                np_dtype = np.float16
            elif dtype == trt.DataType.INT32:
                np_dtype = np.int32
            else:
                np_dtype = np.float32  
            
            size = trt.volume(shape)
            memory_size_mb = size * np_dtype().itemsize / (1024 * 1024)
            
            if memory_size_mb > 1000:
                print(f"경고: 텐서 {name}의 메모리 크기가 매우 큽니다: {memory_size_mb:.2f} MB")

            try:
                host_mem = cuda.pagelocked_empty(size, np_dtype)
                device_mem = cuda.mem_alloc(size * np_dtype().itemsize)
                bindings.append(int(device_mem))

                if mode == trt.TensorIOMode.INPUT:
                    input_buffers[name] = {"host": host_mem, "device": device_mem, "shape": shape, "dtype": np_dtype}
                    print(f"입력 버퍼 할당: {name}, 형태: {shape}, 데이터 타입: {np_dtype}")
                else:
                    output_buffers[name] = {"host": host_mem, "device": device_mem, "shape": shape, "dtype": np_dtype}
                    print(f"출력 버퍼 할당: {name}, 형태: {shape}, 데이터 타입: {np_dtype}")

            except (cuda.MemoryError, pycuda._driver.MemoryError) as e:
                print(f"메모리 할당 오류: {str(e)}")
                raise
    
    except Exception as e:
        print(f"버퍼 할당 중 오류 발생: {str(e)}")
        raise
    
    return input_buffers, output_buffers, bindings

def preprocess_image(image_path, input_shape):
    """입력 이미지 전처리"""
    # 이미지 로드 및 리사이즈
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # BGR -> RGB 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기 조정
    image = cv2.resize(image, (input_shape[3], input_shape[2]))
    
    # [0-255] -> [0-1] 스케일 조정
    image = image.astype(np.float32) / 255.0
    
    # 채널별 정규화 (ImageNet 평균, 표준편차 사용)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # HWC -> CHW (채널 우선 형식으로 변환)
    image = image.transpose((2, 0, 1))
    
    # 배치 차원 추가
    image = np.expand_dims(image, axis=0)
    
    return image


def infer(engine, input_data, input_buffers, output_buffers, bindings):
    """TensorRT 엔진을 사용하여 추론 실행"""
    stream = cuda.Stream()
    context = engine.create_execution_context()

    try:
        # ✅ 입력 텐서 이름 가져오기
        input_name = list(input_buffers.keys())[0]  # 첫 번째 입력 텐서
        input_shape = input_buffers[input_name]["shape"]

        # ✅ 동적 입력 크기 설정
        if hasattr(context, "set_input_shape"):
            print("Using set_input_shape()")
            context.set_input_shape(input_name, input_shape)  
        else:
            print("Using set_binding_shape()")
            context.set_binding_shape(0, input_shape)  # 기존 방식 사용

        # ✅ 입력 데이터 복사
        for name, buffer in input_buffers.items():
            np.copyto(buffer["host"], input_data.ravel())
            cuda.memcpy_htod_async(buffer["device"], buffer["host"], stream)

        # ✅ 강제적으로 `execute_async_v2()` 사용하여 오류 방지!
        print("Using execute_async_v2()")
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # ✅ 결과를 디바이스에서 호스트로 복사
        for name, buffer in output_buffers.items():
            cuda.memcpy_dtoh_async(buffer["host"], buffer["device"], stream)

        stream.synchronize()

        # ✅ 출력 결과 반환
        outputs = {}
        for name, buffer in output_buffers.items():
            outputs[name] = buffer["host"].reshape(buffer["shape"])

        return outputs

    except Exception as e:
        print(f"추론 실행 중 오류 발생: {str(e)}")
        raise

    finally:
        del context

# 메모리 정보 출력 함수
def print_gpu_memory_info():
    try:
        free, total = cuda.mem_get_info()
        free_mb = free / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        used_mb = total_mb - free_mb
        print(f"GPU 메모리 정보: 사용 중 {used_mb:.2f} MB / 전체 {total_mb:.2f} MB (여유: {free_mb:.2f} MB, {free/total*100:.1f}%)")
    except Exception as e:
        print(f"GPU 메모리 정보를 가져올 수 없습니다: {str(e)}")

# 메인 함수
def main(image_path):
    # 메모리 상태 확인
    print_gpu_memory_info()
    
    # 엔진 로드
    print(f"TensorRT 엔진 로드 중: {engine_path}")
    engine = load_engine(engine_path)
    
    # 엔진 정보 출력
    print("\n==== TensorRT 엔진 정보 ====")
    print(f"엔진에서 I/O 텐서 개수: {engine.num_io_tensors}")
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        
        mode_str = "입력" if mode == trt.TensorIOMode.INPUT else "출력"
        print(f"텐서 {i}: 이름={name}, 모드={mode_str}, 형태={shape}, 데이터 타입={dtype}")
    
    try:
        # 버퍼 할당
        print("\n메모리 버퍼 할당 중...")
        input_buffers, output_buffers, bindings = allocate_buffers(engine)
        
        # 입력 텐서 이름과 형태 가져오기
        input_name = engine.get_tensor_name(0)
        input_shape = input_buffers[input_name]["shape"]
        
        # 이미지 전처리
        print(f"\n이미지 전처리 중: {image_path}")
        input_data = preprocess_image(image_path, input_shape)
        
        # 추론 실행
        print("\n추론 실행 중...")
        start_time = time.time()
        outputs = infer(engine, input_data, input_buffers, output_buffers, bindings)
        inference_time = time.time() - start_time
        
        # 결과 출력
        print(f"\n추론 완료. 추론 시간: {inference_time:.3f}초")
        for name, output in outputs.items():
            print(f"출력 '{name}' 형태: {output.shape}")
            print(f"출력 값 요약: 최소={output.min():.6f}, 최대={output.max():.6f}, 평균={output.mean():.6f}")
        
        return outputs
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
    
    finally:
        # 메모리 정리
        gc.collect()
        cuda.Context.synchronize()
        print_gpu_memory_info()

# 테스트 이미지 경로 (실제 경로로 변경해주세요)
image_path = "assets/289.jpg"

# 스크립트가 직접 실행될 때만 메인 함수 호출
if __name__ == "__main__":
    if not os.path.exists(image_path):
        print(f"오류: 테스트 이미지 파일이 존재하지 않습니다. 올바른 경로를 지정해주세요: {image_path}")
    else:
        outputs = main(image_path)