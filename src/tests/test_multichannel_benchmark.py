"""FHD 멀티채널 GPU 벤치마크 — ONNX / TRT FP32 / TRT Mixed 비교.

목적:
  - 3개 백엔드 동일 조건 멀티채널 성능 비교
  - 채널 수 증가에 따른 지연시간, FPS, 처리량, GPU/CPU 메모리 측정
  - OOM 한계까지 자동 탐색

사용법:
  # 터미널 출력 (기본)
  python -m src.tests.test_multichannel_benchmark

  # 마크다운 보고서 생성
  python -m src.tests.test_multichannel_benchmark --markdown result_test.md

  # 특정 백엔드만
  python -m src.tests.test_multichannel_benchmark --models onnx trt_mixed

  # JSON 저장
  python -m src.tests.test_multichannel_benchmark --save results/benchmark.json
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import InferenceConfig
from src.onnx import ClipEBCOnnx
from src.trt import ClipEBCTensorRT

# ── 상수 ──
FHD_H, FHD_W = 1080, 1920

MODEL_PATHS = {
    "onnx":      "assets/CLIP_EBC_nwpu_rmse_onnx.onnx",
    "trt_fp32":  "assets/CLIP_EBC_nwpu_rmse_tensorrt.engine",
    "trt_mixed": "assets/CLIP_EBC_nwpu_rmse_tensorrt_mixed.engine",
}

# OOM 시점까지 자동 확장되는 채널 시퀀스
CHANNEL_SEQUENCE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


# ── 유틸리티 ──
def gpu_info() -> Dict[str, float]:
    if _NVML:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "used_mb": info.used / 1024**2,
            "free_mb": info.free / 1024**2,
            "total_mb": info.total / 1024**2,
        }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        vals = [v.strip() for v in result.stdout.strip().split(",")]
        return {"used_mb": float(vals[0]), "free_mb": float(vals[1]), "total_mb": float(vals[2])}
    except Exception:
        return {"used_mb": -1, "free_mb": -1, "total_mb": -1}


def gpu_name() -> str:
    if _NVML:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetName(handle)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


def cpu_mem_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def tiles_for_resolution(h: int, w: int, ws: int = 224) -> int:
    rows = max(1, int(np.ceil((h - ws) / ws)) + 1)
    cols = max(1, int(np.ceil((w - ws) / ws)) + 1)
    return rows * cols


def load_model(name: str):
    path = MODEL_PATHS.get(name)
    if not path or not os.path.exists(path):
        return None
    cfg = InferenceConfig(model_path=path)
    if name == "onnx":
        return ClipEBCOnnx(cfg)
    return ClipEBCTensorRT(cfg)


def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ── 단일 백엔드 벤치마크 (OOM까지 자동 탐색) ──
def benchmark_backend(model_name: str, channels: List[int]) -> List[Dict]:
    """단일 백엔드 채널별 벤치마크. OOM 발생 시 자동 중단."""
    clean_gpu()
    model = load_model(model_name)
    if model is None:
        print(f"  [{model_name}] SKIP -- 엔진 파일 없음")
        return []

    tiles_per_ch = tiles_for_resolution(FHD_H, FHD_W)

    # PIA와 동일: 모델 로드 후 단일 이미지 warmup 3회
    warmup_img = [np.zeros((FHD_H, FHD_W, 3), dtype=np.uint8)]
    for _ in range(3):
        model.predict(warmup_img)

    base_gpu = gpu_info()["used_mb"]

    print(f"\n  [{model_name}] 모델 로드 완료 -- 기본 GPU: {base_gpu:.0f} MB")
    print(f"  {'채널':>6s} {'총타일':>8s} {'지연(ms)':>10s} {'채널FPS':>10s} {'처리량(img/s)':>14s} "
          f"{'GPU(MB)':>10s} {'GPU증분':>12s} {'CPU(MB)':>10s}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*14} {'-'*10} {'-'*12} {'-'*10}")

    results = []

    for n_ch in channels:
        images = [np.zeros((FHD_H, FHD_W, 3), dtype=np.uint8)] * n_ch
        total_tiles = tiles_per_ch * n_ch

        try:
            t0 = time.perf_counter()
            model.predict(images)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            gpu_used = gpu_info()["used_mb"]
            gpu_delta = gpu_used - base_gpu
            cpu_used = cpu_mem_mb()

            ch_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
            throughput = n_ch / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0

            row = {
                "channels": n_ch, "total_tiles": total_tiles,
                "latency_ms": round(elapsed_ms, 1),
                "channel_fps": round(ch_fps, 2),
                "throughput": round(throughput, 1),
                "gpu_mb": round(gpu_used),
                "gpu_delta_mb": round(gpu_delta),
                "cpu_mb": round(cpu_used),
                "status": "OK",
            }
            results.append(row)

            print(f"  {n_ch:>6d} {total_tiles:>8d} {elapsed_ms:>9.0f}ms {ch_fps:>9.2f} {throughput:>13.1f} "
                  f"{gpu_used:>9.0f}MB {'+' + str(round(gpu_delta)):>11s}MB {cpu_used:>9.0f}MB")

        except Exception:
            cpu_used = cpu_mem_mb()
            oom_type = "GPU OOM" if model_name == "onnx" else "CPU OOM"
            print(f"  {n_ch:>6d} {total_tiles:>8d}    {oom_type}")
            results.append({
                "channels": n_ch, "total_tiles": total_tiles,
                "latency_ms": None, "channel_fps": None, "throughput": None,
                "gpu_mb": None, "gpu_delta_mb": None, "cpu_mb": round(cpu_used),
                "status": "OOM",
            })
            gc.collect()
            break

        gc.collect()

    del model
    clean_gpu()
    return results


# ── 전체 벤치마크 실행 ──
def run_full_benchmark(model_names: List[str], channels: List[int]) -> Dict[str, List[Dict]]:
    tiles_per_ch = tiles_for_resolution(FHD_H, FHD_W)
    g_info = gpu_info()

    print("=" * 90)
    print(f"  FHD 멀티채널 GPU 벤치마크")
    print(f"  GPU: {gpu_name()}")
    print(f"  총 VRAM: {g_info['total_mb']:.0f} MB")
    print(f"  해상도: FHD ({FHD_W}x{FHD_H})")
    print(f"  채널당 타일 수: {tiles_per_ch}")
    print(f"  테스트 백엔드: {model_names}")
    print(f"  채널 목록: {channels}")
    print(f"  시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    all_results = {}
    for mname in model_names:
        results = benchmark_backend(mname, channels)
        all_results[mname] = results

    return all_results


# ── 터미널 비교 테이블 ──
def print_comparison(all_results: Dict[str, List[Dict]]):
    backends = list(all_results.keys())
    if not backends:
        return

    all_channels = sorted(set(
        r["channels"] for results in all_results.values() for r in results
    ))

    sections = [
        ("지연시간 (ms)", "latency_ms", lambda v: f"{v:>13.0f}ms"),
        ("채널 FPS", "channel_fps", lambda v: f"{v:>14.2f}"),
        ("전체 처리량 (img/s)", "throughput", lambda v: f"{v:>14.1f}"),
        ("GPU 메모리 (MB)", "gpu_mb", lambda v: f"{v:>14,d}"),
        ("GPU 증분 (MB)", "gpu_delta_mb", lambda v: f"+{v:>13,d}"),
    ]

    for title, key, fmt in sections:
        print(f"\n{'=' * 90}")
        print(f"  비교 테이블: {title}")
        print("=" * 90)
        header = f"  {'채널':>6s}"
        for b in backends:
            header += f" {b:>14s}"
        print(header)
        print(f"  {'-'*6}" + f" {'-'*14}" * len(backends))

        for ch in all_channels:
            row = f"  {ch:>6d}"
            for b in backends:
                val = next((r for r in all_results[b] if r["channels"] == ch), None)
                if val and val[key] is not None:
                    row += f" {fmt(val[key])}"
                else:
                    status = val["status"] if val else "-"
                    row += f" {'OOM':>14s}" if status == "OOM" else f" {'-':>14s}"
            print(row)

    # ONNX 대비 속도
    if "onnx" in all_results and len(backends) > 1:
        print(f"\n{'=' * 90}")
        print("  속도 비교: ONNX 대비 배율")
        print("=" * 90)
        header = f"  {'채널':>6s}"
        for b in backends:
            if b != "onnx":
                header += f" {b:>14s}"
        print(header)
        print(f"  {'-'*6}" + f" {'-'*14}" * (len(backends) - 1))

        for ch in all_channels:
            onnx_val = next((r for r in all_results["onnx"] if r["channels"] == ch), None)
            if not onnx_val or onnx_val["latency_ms"] is None:
                continue
            row = f"  {ch:>6d}"
            for b in backends:
                if b == "onnx":
                    continue
                val = next((r for r in all_results[b] if r["channels"] == ch), None)
                if val and val["latency_ms"] is not None:
                    row += f" {onnx_val['latency_ms'] / val['latency_ms']:>13.2f}x"
                else:
                    row += f" {'-':>14s}"
            print(row)


# ── 터미널 요약 ──
def print_summary(all_results: Dict[str, List[Dict]]):
    print(f"\n\n{'=' * 90}")
    print("  최종 요약")
    print("=" * 90)

    for backend, results in all_results.items():
        ok_results = [r for r in results if r["status"] == "OK"]
        if not ok_results:
            print(f"\n  [{backend}] 테스트 실패 또는 모델 없음")
            continue

        max_ch = max(r["channels"] for r in ok_results)
        oom_results = [r for r in results if r["status"] == "OOM"]
        oom_ch = oom_results[0]["channels"] if oom_results else None
        avg_throughput = np.mean([r["throughput"] for r in ok_results])
        single_ch = next((r for r in ok_results if r["channels"] == 1), None)
        max_ch_result = next((r for r in ok_results if r["channels"] == max_ch), None)

        print(f"\n  [{backend}]")
        print(f"    최대 처리 가능 채널 수: {max_ch}채널" + (f" ({oom_ch}채널 OOM)" if oom_ch else ""))
        if single_ch:
            print(f"    1채널 지연시간: {single_ch['latency_ms']:.0f} ms ({single_ch['channel_fps']:.2f} fps)")
        if max_ch_result:
            print(f"    {max_ch}채널 지연시간: {max_ch_result['latency_ms']:.0f} ms ({max_ch_result['channel_fps']:.2f} fps/채널)")
        print(f"    평균 전체 처리량: {avg_throughput:.1f} img/s")


# ── 마크다운 보고서 생성 ──
def generate_markdown(all_results: Dict[str, List[Dict]], output_path: str):
    """result_test.md 형식의 마크다운 보고서를 자동 생성."""
    backends = list(all_results.keys())
    tiles_per_ch = tiles_for_resolution(FHD_H, FHD_W)
    g = gpu_info()
    lines = []

    def ln(s=""):
        lines.append(s)

    # ── 헤더 ──
    ln("# CLIP EBC 백엔드 벤치마크 결과")
    ln()
    ln(f"**테스트 날짜**: {datetime.now().strftime('%Y-%m-%d')}")
    ln()
    ln("---")
    ln()

    # ── 1. 환경 ──
    ln("## 1. 테스트 환경")
    ln()
    ln("| 항목 | 내용 |")
    ln("|---|---|")
    ln(f"| GPU | {gpu_name()} ({g['total_mb']/1024:.0f} GB) |")
    ln(f"| 해상도 | FHD ({FHD_W} x {FHD_H}) |")
    ln(f"| 채널당 타일 수 | {tiles_per_ch}개 |")
    ln("| 테스트 방법 | PIA 패키지 동일 조건 -- warmup 없이 1회 측정, zero 이미지 공유 참조, `gc.collect()` 만 사용 |")
    ln()
    ln("---")
    ln()

    # ── 2. 백엔드별 측정 결과 ──
    ln("## 2. 백엔드별 측정 결과")
    ln()

    backend_labels = {"onnx": "ONNX", "trt_fp32": "TRT FP32", "trt_mixed": "TRT Mixed (FP16 + FP32)"}
    for bname in backends:
        results = all_results[bname]
        ln(f"### {backend_labels.get(bname, bname)}")
        ln()
        ln("| 채널 | 총 타일 | 지연시간 | 채널 FPS | 처리량 (img/s) | GPU 사용 (MB) | GPU 증분 (MB) | CPU (MB) | 상태 |")
        ln("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in results:
            if r["status"] == "OK":
                ln(f"| {r['channels']:,} | {r['total_tiles']:,} | {r['latency_ms']:,.0f} ms | {r['channel_fps']:.2f} "
                   f"| {r['throughput']:.1f} | {r['gpu_mb']:,} | +{r['gpu_delta_mb']:,} | {r['cpu_mb']:,} | OK |")
            else:
                oom_type = "GPU OOM" if bname == "onnx" else "CPU OOM"
                ln(f"| {r['channels']:,} | {r['total_tiles']:,} | - | - | - | - | - | - | **{oom_type}** |")
        ln()

    ln("---")
    ln()

    # ── 3. 비교 분석 ──
    ln("## 3. 비교 분석")
    ln()

    all_channels = sorted(set(r["channels"] for res in all_results.values() for r in res))

    # 3-1. 지연시간
    ln("### 3-1. 지연시간 비교")
    ln()
    header = "| 채널 |"
    sep = "|---:|"
    for b in backends:
        header += f" {backend_labels.get(b, b)} |"
        sep += "---:|"
    ln(header)
    ln(sep)
    for ch in all_channels:
        row = f"| {ch:,} |"
        for b in backends:
            val = next((r for r in all_results[b] if r["channels"] == ch), None)
            if val and val["latency_ms"] is not None:
                row += f" {val['latency_ms']:,.0f} ms |"
            elif val and val["status"] == "OOM":
                row += " OOM |"
            else:
                row += " - |"
        ln(row)
    ln()

    # 3-2. 처리량
    ln("### 3-2. 전체 처리량 비교 (img/s)")
    ln()
    ln(header.replace("지연시간", "처리량"))
    ln(sep)
    for ch in all_channels:
        row = f"| {ch:,} |"
        for b in backends:
            val = next((r for r in all_results[b] if r["channels"] == ch), None)
            if val and val["throughput"] is not None:
                row += f" {val['throughput']:.1f} |"
            elif val and val["status"] == "OOM":
                row += " OOM |"
            else:
                row += " - |"
        ln(row)
    # 평균
    row = "| **평균** |"
    for b in backends:
        ok = [r["throughput"] for r in all_results[b] if r["status"] == "OK"]
        row += f" **{np.mean(ok):.1f}** |" if ok else " - |"
    ln(row)
    ln()

    # 3-3. GPU 메모리
    ln("### 3-3. GPU 메모리 비교")
    ln()
    ln("| 항목 |" + "".join(f" {backend_labels.get(b, b)} |" for b in backends))
    ln("|---|" + "---:|" * len(backends))

    # 모델 로드 VRAM
    row = "| 모델 로드 VRAM |"
    for b in backends:
        first_ok = next((r for r in all_results[b] if r["status"] == "OK"), None)
        row += f" {first_ok['gpu_mb']:,} MB |" if first_ok else " - |"
    ln(row)

    # 채널당 증분
    row = "| 채널당 VRAM 증분 |"
    for b in backends:
        ok = [r for r in all_results[b] if r["status"] == "OK"]
        if len(ok) >= 2:
            chs = np.array([r["channels"] for r in ok])
            gpus = np.array([r["gpu_delta_mb"] for r in ok])
            slope = np.polyfit(chs, gpus, 1)[0]
            row += f" ~{round(slope):,} MB/ch |"
        else:
            row += " - |"
    ln(row)

    # 특정 채널 VRAM
    for ch_check in [32, 64, 128, 1024]:
        row = f"| {ch_check:,}ch 총 VRAM |"
        for b in backends:
            val = next((r for r in all_results[b] if r["channels"] == ch_check), None)
            if val and val["gpu_mb"] is not None:
                row += f" {val['gpu_mb']:,} MB ({val['gpu_mb']/1024:.1f} GB) |"
            elif val and val["status"] == "OOM":
                row += " OOM |"
            else:
                row += " - |"
        ln(row)
    ln()

    # 3-4. OOM 한계
    ln("### 3-4. OOM 한계 비교")
    ln()
    ln("| 항목 |" + "".join(f" {backend_labels.get(b, b)} |" for b in backends))
    ln("|---|" + "---|" * len(backends))
    # 최대 채널
    row = "| 최대 처리 가능 채널 |"
    for b in backends:
        ok = [r for r in all_results[b] if r["status"] == "OK"]
        oom = [r for r in all_results[b] if r["status"] == "OOM"]
        max_ch = max(r["channels"] for r in ok) if ok else 0
        oom_ch = oom[0]["channels"] if oom else None
        row += f" {max_ch:,}ch"
        if oom_ch:
            row += f" ({oom_ch:,}ch OOM)"
        row += " |"
    ln(row)
    # OOM 원인
    row = "| OOM 원인 |"
    for b in backends:
        oom = [r for r in all_results[b] if r["status"] == "OOM"]
        if not oom:
            row += " OOM 미발생 |"
        elif b == "onnx":
            row += " GPU 메모리 |"
        else:
            row += " CPU 메모리 (전처리 numpy 배치) |"
    ln(row)
    # OOM 직전 GPU
    row = "| OOM 직전 GPU 사용량 |"
    for b in backends:
        ok = [r for r in all_results[b] if r["status"] == "OK"]
        if ok:
            last = ok[-1]
            row += f" {last['gpu_mb']:,} MB |"
        else:
            row += " - |"
    ln(row)
    # OOM 직전 처리시간
    row = "| OOM 직전 처리 시간 |"
    for b in backends:
        ok = [r for r in all_results[b] if r["status"] == "OK"]
        if ok:
            last = ok[-1]
            row += f" {last['latency_ms']:,.0f} ms ({last['channels']:,}ch) |"
        else:
            row += " - |"
    ln(row)
    ln()

    # 3-5. ONNX 대비 속도
    if "onnx" in all_results and len(backends) > 1:
        ln("### 3-5. ONNX 대비 속도 비교 (배율)")
        ln()
        header = "| 채널 |"
        sep = "|---:|"
        for b in backends:
            if b != "onnx":
                header += f" {backend_labels.get(b, b)} |"
                sep += "---:|"
        ln(header)
        ln(sep)
        for ch in all_channels:
            onnx_val = next((r for r in all_results["onnx"] if r["channels"] == ch), None)
            if not onnx_val or onnx_val["latency_ms"] is None:
                continue
            row = f"| {ch:,} |"
            for b in backends:
                if b == "onnx":
                    continue
                val = next((r for r in all_results[b] if r["channels"] == ch), None)
                if val and val["latency_ms"] is not None:
                    row += f" {onnx_val['latency_ms'] / val['latency_ms']:.2f}x |"
                else:
                    row += " - |"
            ln(row)
        ln()

    ln("---")
    ln()

    # ── 4. 최종 결론 ──
    ln("## 4. 최종 결론")
    ln()
    ln("| 항목 |" + "".join(f" {backend_labels.get(b, b)} |" for b in backends))
    ln("|---|" + "---:|" * len(backends))

    # 1채널 FPS
    row = "| 1채널 FPS |"
    for b in backends:
        val = next((r for r in all_results[b] if r["channels"] == 1 and r["status"] == "OK"), None)
        row += f" {val['channel_fps']:.1f} |" if val else " - |"
    ln(row)

    # 최대 채널
    row = "| 최대 채널 수 |"
    for b in backends:
        ok = [r for r in all_results[b] if r["status"] == "OK"]
        oom = [r for r in all_results[b] if r["status"] == "OOM"]
        max_ch = max(r["channels"] for r in ok) if ok else 0
        oom_ch = oom[0]["channels"] if oom else None
        txt = f"{max_ch:,}ch"
        if oom_ch:
            oom_type = "GPU OOM" if b == "onnx" else "CPU OOM"
            txt += f" ({oom_ch:,}ch {oom_type})"
        row += f" {txt} |"
    ln(row)

    # 평균 처리량
    row = "| 평균 처리량 |"
    for b in backends:
        ok = [r["throughput"] for r in all_results[b] if r["status"] == "OK"]
        row += f" {np.mean(ok):.1f} img/s |" if ok else " - |"
    ln(row)

    # ONNX 대비 속도
    if "onnx" in all_results:
        onnx_ok = [r["throughput"] for r in all_results["onnx"] if r["status"] == "OK"]
        onnx_avg = np.mean(onnx_ok) if onnx_ok else 0
        row = "| ONNX 대비 속도 |"
        for b in backends:
            ok = [r["throughput"] for r in all_results[b] if r["status"] == "OK"]
            avg = np.mean(ok) if ok else 0
            if b == "onnx":
                row += " 1.0x |"
            elif onnx_avg > 0:
                row += f" {avg / onnx_avg:.1f}x |"
            else:
                row += " - |"
        ln(row)

    # OOM 한계 요인
    row = "| OOM 한계 요인 |"
    for b in backends:
        oom = [r for r in all_results[b] if r["status"] == "OOM"]
        if not oom:
            row += " OOM 미발생 |"
        elif b == "onnx":
            row += " GPU 메모리 |"
        else:
            row += " CPU 메모리 |"
    ln(row)
    ln()

    # 기록
    ln(f"> 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ln(f"> 생성 명령: `python -m src.tests.test_multichannel_benchmark --markdown {output_path}`")

    # 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  마크다운 보고서 저장: {output_path}")


# ── JSON 저장 ──
def save_json(all_results: Dict, output_path: str):
    meta = {
        "gpu": gpu_name(),
        "total_vram_mb": round(gpu_info()["total_mb"]),
        "resolution": f"{FHD_W}x{FHD_H}",
        "tiles_per_channel": tiles_for_resolution(FHD_H, FHD_W),
        "timestamp": datetime.now().isoformat(),
    }
    data = {"meta": meta, "results": all_results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 저장: {output_path}")


# ── 메인 ──
def main():
    parser = argparse.ArgumentParser(description="FHD 멀티채널 GPU 벤치마크")
    parser.add_argument("--models", nargs="+", default=["onnx", "trt_fp32", "trt_mixed"],
                        choices=["onnx", "trt_fp32", "trt_mixed"])
    parser.add_argument("--channels", nargs="+", type=int, default=None,
                        help="테스트할 채널 목록 (기본: OOM까지 자동 탐색)")
    parser.add_argument("--save", type=str, default=None,
                        help="JSON 저장 경로 (예: results/benchmark.json)")
    parser.add_argument("--markdown", type=str, default=None,
                        help="마크다운 보고서 저장 경로 (예: result_test.md)")
    args = parser.parse_args()

    available = [m for m in args.models if os.path.exists(MODEL_PATHS.get(m, ""))]
    if not available:
        print("사용 가능한 모델이 없습니다.")
        for k, v in MODEL_PATHS.items():
            print(f"  {k}: {v} -- {'OK' if os.path.exists(v) else 'MISSING'}")
        return

    channels = sorted(args.channels) if args.channels else CHANNEL_SEQUENCE
    all_results = run_full_benchmark(available, channels)

    print_comparison(all_results)
    print_summary(all_results)

    if args.save:
        save_json(all_results, args.save)

    if args.markdown:
        generate_markdown(all_results, args.markdown)

    print(f"\n{'=' * 90}")
    print("  벤치마크 완료")
    print(f"  종료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    import assets
    main()
