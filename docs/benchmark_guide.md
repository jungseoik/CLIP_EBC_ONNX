# 멀티채널 벤치마크 가이드

CCTV 다채널 동시 처리 시나리오에서 ONNX / TRT FP32 / TRT Mixed 백엔드의 성능을 측정하는 도구입니다.

## 측정 항목

| 항목 | 설명 |
|---|---|
| 지연시간 (ms) | 전체 채널 1회 배치 추론 소요 시간 |
| 채널 FPS | `1000 / 지연시간` -- 각 채널이 초당 결과를 받는 횟수 |
| 전체 처리량 (img/s) | `채널수 / (지연시간/1000)` -- 시스템 초당 총 처리 이미지 수 |
| GPU 사용량 (MB) | 추론 후 GPU VRAM 사용량 (pynvml 또는 nvidia-smi) |
| GPU 증분 (MB) | 모델 로드 기준 대비 VRAM 증가분 |
| CPU 사용량 (MB) | 프로세스 RSS 메모리 |
| OOM 한계 | 각 백엔드가 처리 가능한 최대 채널 수 |

## 사용법

### 기본 실행

```bash
# OOM까지 자동 탐색 (채널: 1, 2, 4, 8, ..., 1024)
python -m src.tests.test_multichannel_benchmark
```

### 마크다운 보고서 생성

```bash
# result_test.md로 자동 생성
python -m src.tests.test_multichannel_benchmark --markdown result_test.md

# JSON도 함께 저장
python -m src.tests.test_multichannel_benchmark --markdown result_test.md --save results/benchmark.json
```

### 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--models` | 테스트할 백엔드 | `onnx trt_fp32 trt_mixed` |
| `--channels` | 테스트할 채널 목록 | `1 2 4 8 16 32 64 128 256 512 1024` (OOM 시 중단) |
| `--markdown` | 마크다운 보고서 저장 경로 | 없음 |
| `--save` | JSON 결과 저장 경로 | 없음 |

### 예시

```bash
# TRT Mixed만 테스트
python -m src.tests.test_multichannel_benchmark --models trt_mixed

# 특정 채널만
python -m src.tests.test_multichannel_benchmark --channels 1 16 64 --markdown result_quick.md

# ONNX와 TRT Mixed 비교
python -m src.tests.test_multichannel_benchmark --models onnx trt_mixed --markdown result_compare.md
```

## 테스트 방법론

PIA AI Package의 `test_limit_channel.py`와 동일한 조건으로 측정합니다.

| 항목 | 방식 |
|---|---|
| 이미지 | `np.zeros((1080, 1920, 3), dtype=np.uint8)` 단일 배열을 공유 참조 |
| Warmup | 모델 로드 후 1장 이미지로 3회만 실행, 이후 채널별 warmup 없음 |
| 측정 | 채널별 1회 측정 (`time.perf_counter`) |
| GPU 정리 | `gc.collect()` 만 사용 (`torch.cuda.empty_cache()` 미사용) |
| OOM 처리 | 예외 발생 시 해당 채널에서 중단 |

### `torch.cuda.empty_cache()`를 사용하지 않는 이유

ONNX Runtime은 자체 GPU 메모리 arena를 관리하며, `torch.cuda.empty_cache()`는 PyTorch 전용 캐시만 해제합니다. ONNX 추론 벤치마크에서 이를 반복 호출하면:

1. ONNX Runtime의 arena 메모리는 해제되지 않습니다.
2. PyTorch 캐시만 반복 해제/재할당되어 GPU 메모리 단편화가 발생합니다.
3. 결과적으로 실제보다 낮은 최대 채널 수가 측정됩니다.

## TRT GPU 메모리가 고정되는 이유

TRT 백엔드는 채널 수가 증가해도 GPU VRAM이 거의 변하지 않습니다. 이는 엔진 빌드 시 설정한 `max_batch_size`에 의해 결정됩니다.

### 메커니즘

```
엔진 빌드 시:
  profile.set_shape(input, min=(1,3,224,224), opt=(64,...), max=(1024,...))
  → TRT가 max_batch=1024 기준으로 GPU 메모리를 선점

런타임 추론 시:
  128채널 × 45타일 = 5,760타일
  → 1,024타일씩 6회 청킹하여 추론
  → 동일 GPU 버퍼를 재사용
  → GPU 메모리 증가 없음
```

### 빌드 파라미터 (`src/config.py`)

```python
@dataclass
class TRTBuildConfig:
    max_batch_size: int = 1024   # GPU 버퍼 크기 결정, VRAM에 직결
    opt_batch_size: int = 64     # TRT 최적화 기준 (FHD 45타일에 근접)
    workspace_gb: int = 16       # 빌드 시 워크스페이스
    mixed_precision: bool = True # FP16 + FP32 혼합 (권장)
```

### max_batch_size에 따른 VRAM 변화

| max_batch_size | TRT Mixed VRAM | TRT FP32 VRAM |
|---|---|---|
| 64 | ~0.6 GB | ~1.2 GB |
| 128 | ~0.8 GB | ~1.6 GB |
| 256 | ~1.3 GB | ~3.2 GB |
| 512 | ~2.3 GB | ~6.0 GB |
| 1024 | ~4.4 GB | ~11.9 GB |

> FHD(1920x1080)는 채널당 45타일이므로 `max_batch_size=64`면 단일 채널 처리에 충분합니다. 다만 현재 코드는 전체 채널의 타일을 하나의 배치로 결합한 뒤 청킹하므로, max_batch를 줄여도 멀티채널 처리량에는 영향이 없습니다.

## OOM 한계 특성

| 백엔드 | OOM 원인 | 한계 요인 |
|---|---|---|
| ONNX | **GPU 메모리** | ONNX Runtime이 배치 크기에 비례하여 GPU 중간 텐서를 동적 할당 |
| TRT FP32 | **CPU 메모리** | `predict()` 내부 `np.vstack()`에서 전체 타일 numpy 배치 생성 |
| TRT Mixed | **CPU 메모리** | 동일 (GPU는 고정) |

TRT 백엔드의 CPU OOM을 완화하려면 `predict()` 내부에서 이미지를 청킹하여 전처리하는 방식으로 개선할 수 있습니다 (현재 미구현).

## 관련 테스트

| 스크립트 | 목적 |
|---|---|
| `test_multichannel_benchmark.py` | 멀티채널 성능 + OOM 한계 + 보고서 |
| `test_vram_benchmark.py` | 해상도별 VRAM + 배치 스케일링 |
| `test_accuracy.py` | 3백엔드 정확도 비교 (count, density map corr, MAE) |
