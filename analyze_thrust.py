#!/usr/bin/env python3
"""
Thrust Coefficient (C_T) Analysis Tool
=======================================
ROS2 bag 데이터셋에서 C_T를 추정하고 thrust 응답을 분석합니다.

Usage:
    # Step 데이터셋 분석 (결과 자동 저장)
    python3 analyze_thrust.py --data_dir /path/to/DataSet-Step/DataSet-Step

    # 저장 경로 지정
    python3 analyze_thrust.py -d /path/to/bags -o ./results

    # 저장 없이 화면만 표시
    python3 analyze_thrust.py -d /path/to/bags --no_save

    # 제외 bag 직접 지정
    python3 analyze_thrust.py -d /path/to/bags --skip no_rpm,data7

    # 공칭 C_T, offset 수동 지정
    python3 analyze_thrust.py -d /path/to/bags --ct_nominal 1.465e-7 --offset_gram -9.8218

필요 패키지 자동 설치: rosbags, numpy, matplotlib, scipy
"""

# ── Auto-install ───────────────────────────────────────────────────────────
import subprocess, sys

def _ensure(pkg):
    mod = pkg.replace("-", "_")
    try:
        __import__(mod)
    except ImportError:
        print(f"[setup] pip install {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for _p in ["rosbags", "numpy", "matplotlib", "scipy"]:
    _ensure(_p)
# ──────────────────────────────────────────────────────────────────────────

import os
import argparse
import struct
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.optimize import minimize_scalar

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ══════════════════════════════════════════════════════════════════════════
# 기본 상수 (argparse로 override 가능)
# ══════════════════════════════════════════════════════════════════════════
G            = 9.80665
OFFSET_GRAM  = -9.8218
C_T_NOMINAL  = 1.465e-7
MAD_WIN      = 21
MAD_THRESH   = 10.0

typestore = get_typestore(Stores.ROS2_HUMBLE)


# ══════════════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════════════════════════════════

def parse_rpm(data: bytes) -> int:
    """SingleActualRpm CDR → rpm (int32 at byte offset 20)"""
    return struct.unpack_from('<i', data, 20)[0]


def detect_spikes(g: np.ndarray,
                  window: int   = MAD_WIN,
                  thresh: float = MAD_THRESH) -> np.ndarray:
    """Local-window MAD z-score 기반 spike mask. True = spike."""
    half = window // 2
    n    = len(g)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo    = max(0, i - half)
        hi    = min(n, i + half + 1)
        local = np.concatenate([g[lo:i], g[i+1:hi]])
        if not len(local):
            continue
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        if mad < 1e-6:
            mad = 1.0
        if np.abs(g[i] - med) / (1.4826 * mad) > thresh:
            mask[i] = True
    return mask


def load_dataset(ds_path: str, offset_gram: float = OFFSET_GRAM):
    """
    bag 폴더 읽기 → dict(t, rpm, T, valid, n_spike, name)
    - load cell spike 제거 & 선형 보간
    - RPM = 0 샘플 제외
    """
    p = Path(ds_path)
    db_files = list(p.glob("*.db3"))
    if not db_files:
        return None

    # RPM
    conn = sqlite3.connect(str(db_files[0]))
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    if 'messages' not in [r[0] for r in cur.fetchall()]:
        conn.close(); return None
    cur.execute("SELECT id, name FROM topics")
    tm = {name: tid for tid, name in cur.fetchall()}
    if '/uav/actual_rpm' not in tm:
        conn.close(); return None
    cur.execute("SELECT timestamp, data FROM messages "
                "WHERE topic_id=? ORDER BY timestamp",
                (tm['/uav/actual_rpm'],))
    rpm_rows = [(r[0] * 1e-9, float(parse_rpm(r[1]))) for r in cur.fetchall()]
    conn.close()

    rpm_t = np.array([r[0] for r in rpm_rows])
    rpm_v = np.array([r[1] for r in rpm_rows])

    # Load cell
    thr_data = []
    with Reader(str(p)) as reader:
        tc = [c for c in reader.connections if c.topic == '/load_cell/weight']
        if not tc:
            return None
        for c, ts_ns, raw in reader.messages(connections=tc):
            msg  = typestore.deserialize_cdr(raw, c.msgtype)
            gram = msg.point.z - offset_gram
            thr_data.append((ts_ns * 1e-9, gram))

    thr_t = np.array([r[0] for r in thr_data])
    thr_g = np.array([r[1] for r in thr_data])

    # Spike removal
    sm = detect_spikes(thr_g)
    thr_g_c = thr_g.copy()
    if sm.sum() > 0:
        thr_g_c[sm] = np.interp(thr_t[sm], thr_t[~sm], thr_g[~sm])

    # Sync
    sr = np.argsort(rpm_t); rpm_t = rpm_t[sr]; rpm_v = rpm_v[sr]
    sh = np.argsort(thr_t); thr_t = thr_t[sh]; thr_g_c = thr_g_c[sh]; sm = sm[sh]
    t0 = max(rpm_t[0], thr_t[0]); t1 = min(rpm_t[-1], thr_t[-1])
    m  = (thr_t >= t0) & (thr_t <= t1)

    t     = thr_t[m]
    T_N   = thr_g_c[m] * 1e-3 * G
    rpm   = np.interp(t, rpm_t, rpm_v)
    valid = ~sm[m]
    valid &= (rpm > 500)

    return dict(t=t, rpm=rpm, T=T_N, valid=valid,
                n_spike=int(sm.sum()), name=p.name)


def estimate_CT(segs: dict, drpm_thresh: float = 50.0):
    """
    정상상태 구간(|dω/dt| < drpm_thresh RPM/s)에서
    T = C_T * ω^2 least-squares 추정.
    """
    all_w2, all_T = [], []
    for d in segs.values():
        t = d['t']; rpm = d['rpm']; T = d['T']; v = d['valid']
        drpm_dt = np.abs(np.concatenate([[0], np.diff(rpm) / np.diff(t)]))
        ss = v & (drpm_dt < drpm_thresh)
        all_w2.append(rpm[ss] ** 2)
        all_T.append(T[ss])
    w2  = np.concatenate(all_w2)
    T_a = np.concatenate(all_T)
    C_T_est  = np.dot(w2, T_a) / np.dot(w2, w2)
    residual = T_a - C_T_est * w2
    return C_T_est, w2, T_a, residual


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="C_T Estimation & Thrust Analysis from ROS2 bag files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--data_dir", "-d", required=True,
        help="bag 폴더들이 있는 디렉토리 (data1/, S2000_5000_01/ 등의 부모)"
    )
    parser.add_argument(
        "--out_dir", "-o", default=None,
        help="결과 저장 폴더 (기본: <data_dir>/analysis_results)"
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="파일 저장 없이 plt.show()로 화면 표시"
    )
    parser.add_argument(
        "--skip", default="no_rpm,offset,idle_rpm,data7",
        help="제외할 bag 이름 쉼표 구분 (기본: 'no_rpm,offset,idle_rpm,data7')"
    )
    parser.add_argument(
        "--ct_nominal", type=float, default=C_T_NOMINAL,
        help=f"공칭 C_T [N/RPM^2] (기본: {C_T_NOMINAL:.4e})"
    )
    parser.add_argument(
        "--offset_gram", type=float, default=OFFSET_GRAM,
        help=f"로드셀 영점 오프셋 [gram] (기본: {OFFSET_GRAM})"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    skip_set = set(s.strip() for s in args.skip.split(','))

    if not data_dir.is_dir():
        print(f"[ERROR] 경로를 찾을 수 없습니다: {data_dir}")
        sys.exit(1)

    # 저장 / 표시 설정
    if args.no_save:
        out_dir = None
        import matplotlib
        matplotlib.use('TkAgg')
    else:
        import matplotlib
        matplotlib.use('Agg')
        out_dir = Path(args.out_dir) if args.out_dir else data_dir / "analysis_results"
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터셋 목록 ──────────────────────────────────────────────────────
    datasets = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name not in skip_set
    ])
    if not datasets:
        print(f"[ERROR] {data_dir} 에 bag 폴더가 없습니다.")
        sys.exit(1)

    print(f"\n{'='*62}")
    print(f"  Thrust Analysis")
    print(f"{'='*62}")
    print(f"  data_dir    : {data_dir}")
    print(f"  datasets    : {len(datasets)} bag folders")
    print(f"  C_T nominal : {args.ct_nominal:.4e} N/RPM²")
    print(f"  offset      : {args.offset_gram:.4f} gram")
    print(f"  output      : {'display only' if args.no_save else out_dir}")
    print()

    # ── [1] 로드 ───────────────────────────────────────────────────────────
    print("[1/3] Loading datasets ...")
    segs = {}
    for ds_path in datasets:
        d = load_dataset(str(ds_path), offset_gram=args.offset_gram)
        if d is None:
            print(f"  SKIP  {ds_path.name}")
            continue
        segs[ds_path.name] = d
        print(f"  OK    {ds_path.name:35s}  "
              f"n={len(d['t']):5d}  valid={d['valid'].sum():5d}  "
              f"spikes={d['n_spike']}")

    if not segs:
        print("[ERROR] 유효한 데이터가 없습니다.")
        sys.exit(1)

    # ── [2] C_T 추정 ───────────────────────────────────────────────────────
    print("\n[2/3] Estimating C_T from steady-state ...")
    C_T_est, w2_ss, T_ss, resid_ss = estimate_CT(segs)
    ratio        = C_T_est / args.ct_nominal
    resid_mN     = resid_ss * 1000
    N_total      = sum(d['valid'].sum() for d in segs.values())

    # per-dataset 통계
    per_ds = {}
    for ds, d in segs.items():
        v        = d['valid']
        T_model  = C_T_est * d['rpm'] ** 2
        resid    = (d['T'][v] - T_model[v]) * 1000
        ratio_v  = d['T'][v] / T_model[v]
        per_ds[ds] = dict(
            rmse        = float(np.sqrt(np.mean(resid**2))),
            ratio_mean  = float(ratio_v.mean()),
            ratio_std   = float(ratio_v.std()),
            T_model     = T_model,
        )

    all_resid = np.concatenate([
        (d['T'][d['valid']] - C_T_est * d['rpm'][d['valid']]**2) * 1000
        for d in segs.values()
    ])
    overall_rmse = float(np.sqrt(np.mean(all_resid**2)))

    print(f"  C_T nominal   = {args.ct_nominal:.6e} N/RPM²")
    print(f"  C_T estimated = {C_T_est:.6e} N/RPM²  ({(ratio-1)*100:+.2f}%)")
    print(f"  SS residual   = {resid_mN.mean():.2f} ± {resid_mN.std():.2f} mN")
    print(f"  Overall RMSE  = {overall_rmse:.3f} mN  (N={N_total:,})")

    print("\n  Per-dataset RMSE:")
    for ds, r in per_ds.items():
        print(f"    {ds:35s}  RMSE={r['rmse']:7.2f} mN  "
              f"T/T_model={r['ratio_mean']:.4f}±{r['ratio_std']:.4f}")

    # ── [3] 플롯 ───────────────────────────────────────────────────────────
    print("\n[3/3] Generating plots ...")

    # ── Figure 1: C_T 추정 요약 (4-panel) ────────────────────────────────
    fig1 = plt.figure(figsize=(14, 10))
    gs   = GridSpec(2, 2, figure=fig1, hspace=0.42, wspace=0.35)

    # [0,0] SS scatter
    ax = fig1.add_subplot(gs[0, 0])
    T_model_ss = C_T_est * w2_ss
    lim_max = float(np.percentile(T_ss, 99) * 1.05 * 1000)
    ax.scatter(T_model_ss * 1000, T_ss * 1000,
               s=1, alpha=0.2, color='steelblue', label='SS samples')
    ax.plot([0, lim_max], [0, lim_max], 'r--', lw=2, label='1:1')
    ax.set_xlim([0, lim_max]); ax.set_ylim([0, lim_max])
    ax.set_xlabel('C_T_est · ω² [mN]', fontsize=11)
    ax.set_ylabel('T measured [mN]', fontsize=11)
    ax.set_title('Steady-State: Model vs Measured', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # [0,1] Residual histogram
    ax = fig1.add_subplot(gs[0, 1])
    ax.hist(resid_mN, bins=80, color='steelblue', edgecolor='navy',
            alpha=0.75, density=True)
    ax.axvline(resid_mN.mean(), color='red', lw=2.5, ls='--',
               label=f'mean = {resid_mN.mean():.1f} mN')
    ax.axvline(0, color='gray', lw=1.5, ls=':', label='zero')
    ax.set_xlabel('T - C_T_est·ω² [mN]', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Residual Distribution (SS)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # [1,0] Per-dataset RMSE bar
    ax  = fig1.add_subplot(gs[1, 0])
    ds_names  = list(per_ds.keys())
    rmse_vals = [per_ds[d]['rmse'] for d in ds_names]
    cmap_bar  = plt.cm.tab20(np.linspace(0, 1, len(ds_names)))
    bars = ax.bar(range(len(ds_names)), rmse_vals,
                  color=cmap_bar, edgecolor='gray', width=0.7)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)
    ax.axhline(overall_rmse, color='red', lw=2, ls='--',
               label=f'Overall = {overall_rmse:.1f} mN')
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels(ds_names, rotation=60, fontsize=6.5, ha='right')
    ax.set_ylabel('RMSE [mN]', fontsize=11)
    ax.set_title('RMSE per Dataset', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)

    # [1,1] T/T_model ratio
    ax = fig1.add_subplot(gs[1, 1])
    ratio_means = [per_ds[d]['ratio_mean'] for d in ds_names]
    ratio_stds  = [per_ds[d]['ratio_std']  for d in ds_names]
    ax.bar(range(len(ds_names)), ratio_means, yerr=ratio_stds,
           color=cmap_bar, edgecolor='gray', width=0.7,
           capsize=3, error_kw={'linewidth': 1.2})
    ax.axhline(1.0, color='red', lw=2, ls='--', label='ideal = 1.0')
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels(ds_names, rotation=60, fontsize=6.5, ha='right')
    ax.set_ylabel('T_meas / C_T_est·ω²', fontsize=11)
    ax.set_title('Thrust Ratio per Dataset', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)

    fig1.suptitle(
        f'C_T Estimation Summary\n'
        f'C_T_nominal = {args.ct_nominal:.4e}  →  '
        f'C_T_est = {C_T_est:.4e} N/RPM²  ({(ratio-1)*100:+.1f}%)  '
        f'|  Overall RMSE = {overall_rmse:.1f} mN',
        fontsize=12, fontweight='bold')

    # ── Figure 2: Time-series (최대 6개) ─────────────────────────────────
    rep_list  = list(segs.keys())[:min(6, len(segs))]
    ncols     = 2
    nrows     = (len(rep_list) + 1) // 2
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes2_flat  = np.array(axes2).flatten() if len(rep_list) > 1 else [axes2]

    for ax, ds in zip(axes2_flat, rep_list):
        d       = segs[ds]
        t_rel   = d['t'] - d['t'][0]
        T       = d['T'];  T_model = per_ds[ds]['T_model'];  rpm = d['rpm']
        v       = d['valid']

        ax2 = ax.twinx()
        ax.plot(t_rel, T * 1000,         color='#1565C0', lw=1.0,
                alpha=0.85, label='T measured')
        ax.plot(t_rel, T_model * 1000,   color='#F44336', lw=1.3, ls='--',
                alpha=0.85, label='C_T_est·ω²')
        if (~v).sum() > 0:
            ax.scatter(t_rel[~v], T[~v] * 1000, color='orange', s=40,
                       zorder=6, marker='x', linewidths=2, label='Spike')
        ax2.plot(t_rel, rpm, color='gray', lw=0.6, alpha=0.35)
        ax2.set_ylabel('RPM', color='gray', fontsize=8)
        ax.set_title(f'{ds}  |  RMSE={per_ds[ds]["rmse"]:.1f} mN',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel('Thrust [mN]', fontsize=9)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    for ax in axes2_flat[len(rep_list):]:
        ax.set_visible(False)
    fig2.suptitle(f'Thrust Time Series  |  C_T_est = {C_T_est:.4e} N/RPM²',
                  fontsize=12, fontweight='bold')
    fig2.tight_layout()

    # ── Figure 3: RPM vs Thrust scatter ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    cmap_sc = plt.cm.tab20(np.linspace(0, 1, len(segs)))
    for (ds, d), col in zip(segs.items(), cmap_sc):
        v = d['valid']
        ax3.scatter(d['rpm'][v], d['T'][v] * 1000,
                    s=1.5, alpha=0.3, color=col, label=ds)
    rpm_line = np.linspace(
        min(d['rpm'].min() for d in segs.values()) * 0.95,
        max(d['rpm'].max() for d in segs.values()) * 1.02,
        300)
    ax3.plot(rpm_line, args.ct_nominal * rpm_line**2 * 1000,
             'k-', lw=2.5, label=f'C_T_nominal ({args.ct_nominal:.4e})')
    ax3.plot(rpm_line, C_T_est * rpm_line**2 * 1000,
             'r--', lw=2.5, label=f'C_T_est     ({C_T_est:.4e})')
    ax3.set_xlabel('RPM', fontsize=12)
    ax3.set_ylabel('Thrust [mN]', fontsize=12)
    ax3.set_title('RPM vs Measured Thrust — All Datasets',
                  fontsize=12, fontweight='bold')
    ax3.legend(markerscale=8, fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    # ── 저장 or 표시 ──────────────────────────────────────────────────────
    if out_dir:
        p1 = out_dir / "01_CT_estimation_summary.png"
        p2 = out_dir / "02_thrust_timeseries.png"
        p3 = out_dir / "03_rpm_vs_thrust.png"
        fig1.savefig(p1, dpi=600); print(f"  Saved: {p1.name}")
        fig2.savefig(p2, dpi=600); print(f"  Saved: {p2.name}")
        fig3.savefig(p3, dpi=600); print(f"  Saved: {p3.name}")
        plt.close('all')

        # CSV
        csv_path = out_dir / "CT_results.csv"
        with open(csv_path, 'w') as f:
            f.write("dataset,n_samples,n_valid,n_spike,"
                    "rmse_mN,ratio_mean,ratio_std\n")
            for ds, d in segs.items():
                r = per_ds[ds]
                f.write(f"{ds},{len(d['t'])},{d['valid'].sum()},{d['n_spike']},"
                        f"{r['rmse']:.4f},{r['ratio_mean']:.6f},{r['ratio_std']:.6f}\n")
        print(f"  Saved: {csv_path.name}")
    else:
        plt.tight_layout()
        plt.show()

    # ── 최종 요약 출력 ─────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*62}")
    print(f"  C_T nominal   = {args.ct_nominal:.6e} N/RPM²")
    print(f"  C_T estimated = {C_T_est:.6e} N/RPM²  ({(ratio-1)*100:+.2f}%)")
    print(f"  SS residual   = {resid_mN.mean():.2f} ± {resid_mN.std():.2f} mN")
    print(f"  Overall RMSE  = {overall_rmse:.3f} mN  (N={N_total:,})")
    print(f"\n  ★ C_T_est = {C_T_est:.6e} N/RPM²")
    print(f"    Thrust responds instantaneously to ω²")
    print(f"    (fc >> 50 Hz — not identifiable at 100 Hz sampling)")
    if out_dir:
        print(f"\n  Results → {out_dir}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
