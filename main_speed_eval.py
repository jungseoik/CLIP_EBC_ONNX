import time
import numpy as np
from tqdm import tqdm

# 성능 비교 함수
def benchmark_model(model_name, model_instance, image_path, num_runs=10):
    times = []
    counts = []
    
    print(f"\n{model_name} 성능 테스트 ({num_runs}회 실행):")
    
    for i in tqdm(range(num_runs)):
        start_time = time.time()
        count = model_instance.predict(image_path)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        counts.append(count)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_count = np.mean(counts)
    
    print(f"평균 실행 시간: {avg_time:.4f}초 (±{std_time:.4f})")
    print(f"평균 예측 인원: {avg_count:.2f}명")
    
    return {
        'model': model_name,
        'avg_time': avg_time,
        'std_time': std_time,
        'avg_count': avg_count,
        'all_times': times
    }

# 테스트할 이미지 경로
image_path = 'assets/289.jpg'
num_runs = 10

# 결과 저장
results = []

# 1. PyTorch 모델 테스트
print("PyTorch 모델 로딩 중...")
from custom.clip_ebc import ClipEBC
model_pytorch = ClipEBC()
results.append(benchmark_model("PyTorch", model_pytorch, image_path, num_runs))

# 2. ONNX 모델 테스트
print("ONNX 모델 로딩 중...")
from custom.clip_ebc_onnx import ClipEBCOnnx
model_onnx = ClipEBCOnnx()
results.append(benchmark_model("ONNX", model_onnx, image_path, num_runs))

# 3. TensorRT 모델 테스트
print("TensorRT 모델 로딩 중...")
from custom.clip_ebc_tensorrt import ClipEBCTensorRT
model_tensorrt = ClipEBCTensorRT(engine_path="assets/CLIP_EBC_nwpu_rmse_tensorrt.engine")
results.append(benchmark_model("TensorRT", model_tensorrt, image_path, num_runs))

# 결과 요약 및 비교
print("\n성능 비교 요약:")
print("-" * 70)
print(f"{'모델':10} | {'평균 시간(초)':15} | {'속도 향상(PyTorch 대비)':25} | {'예측 인원':10}")
print("-" * 70)

baseline_time = results[0]['avg_time']  # PyTorch 시간을 기준으로 설정

for result in results:
    speedup = baseline_time / result['avg_time']
    print(f"{result['model']:10} | {result['avg_time']:.4f} ±{result['std_time']:.4f} | {speedup:.2f}x {(speedup-1)*100:.1f}% 더 빠름 | {result['avg_count']:.2f}")
# 그래프로 시각화
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 영어 글꼴 사용
    
    # 실행 시간 비교 
    plt.figure(figsize=(10, 6))
    labels = [r['model'] for r in results]
    times = [r['avg_time'] for r in results]
    errors = [r['std_time'] for r in results]
    
    plt.bar(labels, times, yerr=errors, capsize=10)
    plt.ylabel('Average Inference Time (sec)')
    plt.title('Inference Time Comparison by Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 값 표시
    for i, v in enumerate(times):
        plt.text(i, v + errors[i] + 0.01, f"{v:.4f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    print("\nPerformance comparison graph has been saved to 'model_performance_comparison.png'")
except Exception as e:
    print(f"\nGraph generation error: {e}")
    print("For graph creation, matplotlib is required.")