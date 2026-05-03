#!/usr/bin/env python3
"""
Load Cell Spike Removal & Interpolation
========================================
- Outlier detection : local-window MAD (z-score > 10)
- Removal           : 검출된 spike 샘플 제거
- Interpolation     : 선형 보간 (numpy.interp)
- 결과 저장         : <스크립트위치>/cleaned/<dataset>_thrust_clean.npz
- 시각화            : before/after 비교 플롯

필요 패키지 자동 설치: rosbags, numpy, matplotlib
"""

# ── Auto-install missing packages ─────────────────────────────────────────
import subprocess, sys

def ensure(pkg):
    mod = pkg.split("[")[0].replace("-", "_")
    try:
        __import__(mod)
    except ImportError:
        print(f"[setup] installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure("rosbags")
ensure("numpy")
ensure("matplotlib")
# ──────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ══════════════════════════════════════════════════════════════════════════
#  ★ 사용자 설정 (여기만 수정하세요)
# ══════════════════════════════════════════════════════════════════════════

# bag 파일들이 있는 최상위 폴더 (data1, data2, ... 폴더들의 부모)
# 기본값: 이 스크립트와 같은 위치의 DataSet/DataSet 폴더
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE  = os.path.join(_HERE, "DataSet")

# 결과 저장 폴더
OUT_DIR = os.path.join(_HERE, "cleaned")

# 분석에서 제외할 bag 이름
SKIP_BAGS = {'offset', 'idle_rpm', 'data7'}

# Load cell 영점 오프셋 [gram]  (offset bag 평균에서 측정: -9.8218)
OFFSET_GRAM = -9.8218

# MAD spike detection 파라미터
MAD_WIN    = 21     # local window size (samples)
MAD_THRESH = 10.0   # z-score threshold

# ══════════════════════════════════════════════════════════════════════════

G = 9.80665  # m/s^2

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isdir(BASE):
    print(f"[ERROR] BASE 경로를 찾을 수 없습니다:\n  {BASE}")
    print("  스크립트 상단의 BASE 변수를 DataSet 폴더 경로로 수정하세요.")
    sys.exit(1)

DATASETS = sorted([d for d in os.listdir(BASE)
                   if d.startswith('data')
                   and os.path.isdir(os.path.join(BASE, d))
                   and d not in SKIP_BAGS])

typestore = get_typestore(Stores.ROS2_HUMBLE)

print(f"BASE    : {BASE}")
print(f"OUT_DIR : {OUT_DIR}")
print(f"Datasets: {DATASETS}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD CELL 읽기
# ═══════════════════════════════════════════════════════════════════════════

def read_loadcell(ds_name: str):
    """(t_sec, gram_offset_corrected) 배열 반환"""
    ds_path = os.path.join(BASE, ds_name)
    data = []
    with Reader(ds_path) as reader:
        tc = [c for c in reader.connections if c.topic == '/load_cell/weight']
        if not tc:
            return np.array([]), np.array([])
        for c, ts_ns, raw in reader.messages(connections=tc):
            msg  = typestore.deserialize_cdr(raw, c.msgtype)
            gram = msg.point.z - OFFSET_GRAM
            data.append((ts_ns * 1e-9, gram))
    if not data:
        return np.array([]), np.array([])
    arr  = np.array(data)
    sort = np.argsort(arr[:, 0])
    return arr[sort, 0], arr[sort, 1]


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPIKE DETECTION  (local-window MAD)
# ═══════════════════════════════════════════════════════════════════════════

def detect_spikes(g: np.ndarray,
                  window: int   = MAD_WIN,
                  thresh: float = MAD_THRESH) -> np.ndarray:
    """
    각 샘플에 대해 주변 window 내 MAD z-score 계산.
    z > thresh 이면 spike로 판정.
    Returns boolean mask (True = spike)
    """
    half       = window // 2
    n          = len(g)
    spike_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        lo    = max(0, i - half)
        hi    = min(n, i + half + 1)
        local = np.concatenate([g[lo:i], g[i+1:hi]])   # 자기 자신 제외
        if len(local) == 0:
            continue
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        if mad < 1e-6:
            mad = 1.0
        z = np.abs(g[i] - med) / (1.4826 * mad)
        if z > thresh:
            spike_mask[i] = True

    return spike_mask


# ═══════════════════════════════════════════════════════════════════════════
# 3. INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════════

def remove_and_interpolate(t: np.ndarray,
                           g: np.ndarray,
                           spike_mask: np.ndarray) -> np.ndarray:
    """
    spike 샘플을 제거하고 원래 타임스탬프 위치에 선형 보간.
    Returns g_clean (same length, spikes replaced)
    """
    good_t = t[~spike_mask]
    good_g = g[~spike_mask]
    if len(good_t) < 2:
        return g.copy()
    g_clean = g.copy()
    g_clean[spike_mask] = np.interp(t[spike_mask], good_t, good_g)
    return g_clean


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

summary = {}

for ds in DATASETS:
    print(f"── {ds} ──")
    t, g = read_loadcell(ds)

    if len(t) == 0:
        print("  [SKIP] load cell 데이터 없음\n")
        continue

    spike_mask = detect_spikes(g)
    n_spike    = spike_mask.sum()
    print(f"  samples  : {len(t)}")
    print(f"  spikes   : {n_spike}  ({100 * n_spike / len(t):.4f}%)")

    if n_spike > 0:
        for si in np.where(spike_mask)[0]:
            half  = MAD_WIN // 2
            local = g[max(0, si-half) : si+half+1]
            local = local[local != g[si]]
            loc_med = np.median(local) if len(local) else float('nan')
            print(f"    idx={si:5d}  t={t[si]:.4f} s  "
                  f"g={g[si]:+10.2f} gram  (local_med={loc_med:.2f})")

    g_clean = remove_and_interpolate(t, g, spike_mask)
    T_clean = g_clean * 1e-3 * G

    print(f"  gram raw   : [{g.min():.2f}, {g.max():.2f}]")
    print(f"  gram clean : [{g_clean.min():.2f}, {g_clean.max():.2f}]")

    save_path = os.path.join(OUT_DIR, f"{ds}_thrust_clean.npz")
    np.savez(save_path,
             t=t, gram_raw=g, gram_clean=g_clean,
             thrust_N=T_clean, spike_mask=spike_mask)
    print(f"  saved -> {save_path}\n")

    summary[ds] = dict(n=len(t), n_spike=n_spike,
                       t=t, g=g, g_clean=g_clean, spike_mask=spike_mask)


# ═══════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

spiked_ds = {k: v for k, v in summary.items() if v['n_spike'] > 0}

# ── Plot A: before/after for datasets WITH spikes ───────────────────────
if spiked_ds:
    fig, axes = plt.subplots(len(spiked_ds), 1,
                              figsize=(13, 4.5 * len(spiked_ds)), squeeze=False)
    for ax_row, (ds, res) in zip(axes, spiked_ds.items()):
        ax = ax_row[0]
        t_ = res['t'] - res['t'][0]
        sm = res['spike_mask']

        ax.plot(t_, res['g'],       color='#90CAF9', lw=0.8, label='Raw (offset corrected)')
        ax.plot(t_, res['g_clean'], color='#1565C0', lw=1.2, label='Cleaned (interpolated)')
        ax.scatter(t_[sm], res['g'][sm],
                   color='red',  s=60, zorder=5, marker='x',
                   linewidths=2, label=f'Spike ({sm.sum()})')
        ax.scatter(t_[sm], res['g_clean'][sm],
                   color='lime', s=40, zorder=6, marker='o',
                   label='Interpolated value')
        ax.set_title(f"{ds}  —  {sm.sum()} spike(s) removed & interpolated",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Gram (offset corrected)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Load Cell Spike Removal & Interpolation", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "spike_removal_overview.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"Saved: {p}")

# ── Plot B: zoom-in around each spike ───────────────────────────────────
for ds, res in spiked_ds.items():
    t_full        = res['t'] - res['t'][0]
    spike_indices = np.where(res['spike_mask'])[0]

    fig, axes = plt.subplots(len(spike_indices), 1,
                              figsize=(11, 3.5 * len(spike_indices)), squeeze=False)
    for ax_row, si in zip(axes, spike_indices):
        ax  = ax_row[0]
        lo  = max(0, si - 30)
        hi  = min(len(res['g']), si + 30)
        t_w = t_full[lo:hi]

        ax.plot(t_w, res['g'][lo:hi],       'o-', color='#90CAF9', lw=1.0, ms=4, label='Raw')
        ax.plot(t_w, res['g_clean'][lo:hi], 's-', color='#1565C0', lw=1.5, ms=4, label='Cleaned')
        ax.scatter(t_full[si:si+1], res['g'][si:si+1],
                   color='red',  s=120, zorder=6, marker='x',
                   linewidths=2.5, label=f'Spike  g={res["g"][si]:.1f} g')
        ax.scatter(t_full[si:si+1], res['g_clean'][si:si+1],
                   color='lime', s=80, zorder=7, marker='D',
                   label=f'Interp  g={res["g_clean"][si]:.1f} g')
        ax.set_title(f"{ds}  spike idx={si}  t={t_full[si]:.3f} s")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Gram")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{ds} — Spike Zoom-in", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"{ds}_spike_zoom.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"Saved: {p}")

# ── Plot C: all datasets overview (clean) ───────────────────────────────
if summary:
    n_ds  = len(summary)
    ncols = 2
    nrows = (n_ds + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes_flat  = axes.flatten() if n_ds > 1 else [axes]

    for ax, (ds, res) in zip(axes_flat, summary.items()):
        t_rel = res['t'] - res['t'][0]
        ax.plot(t_rel, res['g_clean'], lw=0.8, color='#1565C0', alpha=0.85)
        if res['n_spike'] > 0:
            sm = res['spike_mask']
            ax.scatter(t_rel[sm], res['g_clean'][sm],
                       color='lime', s=30, zorder=5, label='Interpolated')
            ax.legend(fontsize=8)
        ax.set_title(f"{ds}  (spikes={res['n_spike']})", fontsize=10)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Gram (clean)")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_ds:]:
        ax.set_visible(False)

    plt.suptitle("All Datasets — Cleaned Load Cell Data", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "all_datasets_clean.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  SPIKE REMOVAL SUMMARY")
print("=" * 55)
print(f"  {'Dataset':<10} {'Samples':>8} {'Spikes':>8} {'%':>8}")
print(f"  {'-' * 38}")
for ds, res in summary.items():
    pct = 100 * res['n_spike'] / res['n'] if res['n'] > 0 else 0.0
    print(f"  {ds:<10} {res['n']:>8} {res['n_spike']:>8} {pct:>7.4f}%")
print("=" * 55)
print(f"\nCleaned .npz -> {OUT_DIR}/")
print("Fields: t, gram_raw, gram_clean, thrust_N, spike_mask")