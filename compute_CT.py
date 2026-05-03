#!/usr/bin/env python3
"""
C_T Estimation from Fixed-RPM Load Cell Measurements
=====================================================
각 RPM에서 정상상태로 측정한 로드셀 데이터로 C_T [N/RPM²]를 추정합니다.

T = C_T * omega^2  (least-squares fit, no intercept)

Usage:
    python3 compute_CT.py --data_dir /path/to/Bag_file_with_prop_gauard
    python3 compute_CT.py --data_dir /path/to/bags --no_save
    python3 compute_CT.py --data_dir /path/to/bags --out_dir ./results

Directory structure expected:
    <data_dir>/
        initial_weight/   ← tare bag (motor OFF)
        1200RPM/          ← bag name must contain RPM value
        2000RPM/
        ...
        7500RPM/

필요 패키지 자동 설치: rosbags, numpy, matplotlib, scipy
"""

# ── Auto-install ────────────────────────────────────────────────────────
import subprocess, sys

def _ensure(pkg):
    try: __import__(pkg.replace("-","_"))
    except ImportError:
        print(f"[setup] pip install {pkg} ...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,
                               "--break-system-packages","-q"])

for _p in ["rosbags","numpy","matplotlib","scipy"]:
    _ensure(_p)
# ───────────────────────────────────────────────────────────────────────

import argparse
import struct
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.optimize import curve_fit

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

G         = 9.81      # m/s²
MAD_SIGMA = 5.0          # spike rejection threshold (MAD z-score)

typestore = get_typestore(Stores.ROS2_HUMBLE)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_rpm(data: bytes) -> int:
    return struct.unpack_from('<i', data, 20)[0]


def mad_clean(arr: np.ndarray, sigma: float = MAD_SIGMA):
    """MAD 기반 spike 제거. (clean_array, mask) 반환"""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-9: mad = 1e-9
    mask = np.abs(arr - med) / (1.4826 * mad) < sigma
    return arr[mask], mask


def load_bag(ds_path: str):
    """
    bag 폴더에서 (load_cell_grams, rpm_array) 반환.
    spike는 MAD로 제거.
    """
    p = Path(ds_path)
    db = list(p.glob("*.db3"))[0]
    conn = sqlite3.connect(str(db)); cur = conn.cursor()
    cur.execute("SELECT id,name FROM topics")
    tm = {n:i for i,n in cur.fetchall()}; conn.close()

    # Load cell [gram]
    thr_raw = []
    with Reader(str(p)) as reader:
        tc = [c for c in reader.connections if c.topic == '/load_cell/weight']
        if not tc: return None, None
        for c, ts, raw in reader.messages(connections=tc):
            msg = typestore.deserialize_cdr(raw, c.msgtype)
            thr_raw.append(msg.point.z)
    thr_raw = np.array(thr_raw)
    thr_clean, _ = mad_clean(thr_raw)

    # RPM
    rpm = np.array([])
    if '/uav/actual_rpm' in tm:
        conn = sqlite3.connect(str(db)); cur = conn.cursor()
        cur.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
                    (tm['/uav/actual_rpm'],))
        rpm = np.array([float(parse_rpm(r[0])) for r in cur.fetchall()])
        conn.close()

    return thr_clean, rpm


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="C_T Estimation from fixed-RPM load cell bags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--data_dir", "-d", required=True,
                        help="initial_weight/ 와 <N>RPM/ 폴더가 있는 디렉토리")
    parser.add_argument("--out_dir",  "-o", default=None,
                        help="결과 저장 폴더 (기본: <data_dir>/ct_results)")
    parser.add_argument("--no_save",  action="store_true",
                        help="저장 없이 plt.show() 표시")
    parser.add_argument("--mad_sigma", type=float, default=MAD_SIGMA,
                        help=f"spike 제거 MAD z-score (기본: {MAD_SIGMA})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"[ERROR] 경로 없음: {data_dir}"); sys.exit(1)

    if args.no_save:
        import matplotlib; matplotlib.use('TkAgg')
        out_dir = None
    else:
        import matplotlib; matplotlib.use('Agg')
        out_dir = Path(args.out_dir) if args.out_dir else data_dir / "ct_results"
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  C_T Estimation  —  Prop Guard Dataset")
    print(f"{'='*58}")
    print(f"  data_dir : {data_dir}")
    print(f"  output   : {'display only' if args.no_save else out_dir}\n")

    # ── [1] Tare ──────────────────────────────────────────────────────
    tare_path = data_dir / "initial_weight"
    if not tare_path.exists():
        print("[ERROR] initial_weight/ 폴더를 찾을 수 없습니다."); sys.exit(1)

    tare_raw, _ = load_bag(str(tare_path))
    tare_gram   = tare_raw.mean()
    print(f"[1/3] Tare (initial_weight)")
    print(f"  raw  : mean={np.array([t for t in tare_raw]).mean() if len(tare_raw) else 0:.4f} g  "
          f"std={tare_raw.std():.4f} g  n={len(tare_raw)}")
    print(f"  tare = {tare_gram:.4f} gram\n")

    # ── [2] Per-RPM 로드 ──────────────────────────────────────────────
    rpm_dirs = sorted(
        [d for d in data_dir.iterdir()
         if d.is_dir() and d.name != 'initial_weight' and 'RPM' in d.name.upper()],
        key=lambda x: int(''.join(filter(str.isdigit, x.name)))
    )
    if not rpm_dirs:
        print("[ERROR] RPM bag 폴더가 없습니다."); sys.exit(1)

    print(f"[2/3] Loading {len(rpm_dirs)} RPM bags ...")
    print(f"  {'RPM_cmd':>8} {'RPM_meas':>9} {'T_mean[N]':>10} "
          f"{'T_std[mN]':>10} {'n_clean':>8}")
    print(f"  {'-'*52}")

    data_rows   = []   # (rpm_cmd, rpm_meas, T_mean_N, T_std_N, n)
    noise_by_rpm = {}

    for ds in rpm_dirs:
        thr_raw, rpm = load_bag(str(ds))
        if thr_raw is None: continue

        rpm_cmd  = int(''.join(filter(str.isdigit, ds.name)))
        # T [N] = (measured_gram - tare_gram) * 1e-3 * G
        T_N      = (thr_raw - tare_gram) * 1e-3 * G
        rpm_meas = float(rpm.mean()) if len(rpm) else float(rpm_cmd)

        data_rows.append((rpm_cmd, rpm_meas, T_N.mean(), T_N.std(), len(T_N)))
        noise_by_rpm[rpm_cmd] = T_N.std() * 1000

        print(f"  {rpm_cmd:8d} {rpm_meas:9.1f} {T_N.mean():10.4f} "
              f"{T_N.std()*1000:10.2f} {len(T_N):8d}")

    rpm_cmd_arr  = np.array([r[0] for r in data_rows], dtype=float)
    rpm_meas_arr = np.array([r[1] for r in data_rows], dtype=float)
    T_arr        = np.array([r[2] for r in data_rows])
    T_std_arr    = np.array([r[3] for r in data_rows])

    # ── [3] C_T 추정 ─────────────────────────────────────────────────
    print(f"\n[3/3] Estimating C_T ...")

    # Method A: OLS (no intercept)  T = C_T * omega^2
    w2      = rpm_meas_arr ** 2
    C_T_ols = np.dot(w2, T_arr) / np.dot(w2, w2)
    T_pred_ols  = C_T_ols * w2
    resid_ols   = T_arr - T_pred_ols
    rmse_ols    = np.sqrt(np.mean(resid_ols**2)) * 1000

    # Method B: WLS (weight = 1/std²)
    weights  = 1.0 / np.where(T_std_arr > 0, T_std_arr**2, 1e-12)
    C_T_wls  = np.dot(weights * w2, T_arr) / np.dot(weights * w2, w2)
    T_pred_wls  = C_T_wls * w2
    resid_wls   = T_arr - T_pred_wls
    rmse_wls    = np.sqrt(np.mean(resid_wls**2)) * 1000

    # Method C: with intercept  T = C_T * omega^2 + b  (linearity check)
    A  = np.column_stack([w2, np.ones_like(w2)])
    x, _, _, _ = np.linalg.lstsq(A, T_arr, rcond=None)
    C_T_int, b_int = x
    T_pred_int = C_T_int * w2 + b_int
    rmse_int   = np.sqrt(np.mean((T_arr - T_pred_int)**2)) * 1000

    print(f"\n  {'Method':<30} {'C_T [N/RPM²]':>16} {'RMSE [mN]':>10} {'note'}")
    print(f"  {'-'*72}")
    print(f"  {'OLS (no intercept)':<30} {C_T_ols:.6e} {rmse_ols:10.2f}")
    print(f"  {'WLS (1/std² weight)':<30} {C_T_wls:.6e} {rmse_wls:10.2f}")
    print(f"  {'OLS + intercept':<30} {C_T_int:.6e} {rmse_int:10.2f}  "
          f"b={b_int*1000:.2f} mN")

    # 최종 권장: OLS (no intercept)
    C_T_final = C_T_ols
    print(f"\n  ★ Recommended C_T = {C_T_final:.6e} N/RPM²")
    print(f"    (OLS, no intercept, RMSE = {rmse_ols:.2f} mN)")

    # hover RPM 역산
    m_hover_list = [2.960, 3.230]
    print(f"\n  Hover RPM estimate (6 rotors):")
    for m in m_hover_list:
        T_h = m * G / 6
        rpm_h = np.sqrt(T_h / C_T_final)
        print(f"    m={m:.3f} kg → T/rotor={T_h:.4f} N → RPM={rpm_h:.0f}")

    # ── Plots ──────────────────────────────────────────────────────────
    rpm_fit  = np.linspace(0, rpm_meas_arr.max()*1.05, 400)
    T_fit_ols = C_T_ols * rpm_fit**2
    T_fit_wls = C_T_wls * rpm_fit**2
    T_fit_int = C_T_int * rpm_fit**2 + b_int

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # [0,0] RPM vs Thrust + fits
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(rpm_meas_arr, T_arr, yerr=T_std_arr*3,
                fmt='o', color='#1565C0', ms=7, lw=1.5, capsize=4,
                label='Measured (mean ± 3σ)', zorder=5)
    ax.plot(rpm_fit, T_fit_ols, 'r-',  lw=2.0, label=f'OLS: {C_T_ols:.4e}')
    ax.plot(rpm_fit, T_fit_wls, 'g--', lw=1.8, label=f'WLS: {C_T_wls:.4e}')
    ax.plot(rpm_fit, T_fit_int, 'm:',  lw=1.8,
            label=f'OLS+b: {C_T_int:.4e} (b={b_int*1000:.1f}mN)')
    ax.set_xlabel('RPM', fontsize=11); ax.set_ylabel('Thrust [N]', fontsize=11)
    ax.set_title('RPM vs Thrust + C_T fits', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [0,1] T vs C_T·omega² scatter (linearity)
    ax = fig.add_subplot(gs[0, 1])
    T_model_pts = C_T_ols * rpm_meas_arr**2
    ax.scatter(T_model_pts, T_arr, s=60, color='steelblue',
               edgecolors='navy', zorder=5, label='Data points')
    lim = [0, max(T_arr)*1.05]
    ax.plot(lim, lim, 'r--', lw=2, label='1:1 line')
    for i, (xp, yp, rpm) in enumerate(zip(T_model_pts, T_arr, rpm_cmd_arr)):
        ax.annotate(f'{int(rpm)}', (xp, yp), textcoords='offset points',
                    xytext=(4, 3), fontsize=7, color='gray')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('C_T_OLS · ω² [N]', fontsize=11)
    ax.set_ylabel('T measured [N]', fontsize=11)
    ax.set_title('Linearity Check: Model vs Measured', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # [1,0] Residuals
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(rpm_cmd_arr, resid_ols*1000, width=180,
           color=['#EF9A9A' if r < 0 else '#A5D6A7' for r in resid_ols],
           edgecolor='gray', label='OLS residual')
    ax.bar(rpm_cmd_arr+200, resid_wls*1000, width=180,
           color=['#FF5252' if r < 0 else '#00C853' for r in resid_wls],
           edgecolor='gray', alpha=0.7, label='WLS residual')
    ax.axhline(0, color='k', lw=1.5)
    ax.set_xlabel('RPM', fontsize=11)
    ax.set_ylabel('Residual T - C_T·ω² [mN]', fontsize=11)
    ax.set_title('Residuals per RPM point', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)

    # [1,1] Load cell noise (std) per RPM
    ax = fig.add_subplot(gs[1, 1])
    rpms_n = list(noise_by_rpm.keys())
    stds_n = [noise_by_rpm[r] for r in rpms_n]
    bars = ax.bar(range(len(rpms_n)), stds_n, color=plt.cm.viridis(
        np.linspace(0.2, 0.9, len(rpms_n))), edgecolor='gray', width=0.7)
    for bar, v in zip(bars, stds_n):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(len(rpms_n)))
    ax.set_xticklabels([f'{r}' for r in rpms_n], rotation=45, fontsize=8)
    ax.set_xlabel('RPM', fontsize=11)
    ax.set_ylabel('Load cell std [mN]', fontsize=11)
    ax.set_title('Thrust Noise (std) per RPM', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(
        f'C_T Estimation  —  With Prop Guard\n'
        f'C_T (OLS) = {C_T_ols:.6e} N/RPM²  |  '
        f'C_T (WLS) = {C_T_wls:.6e} N/RPM²  |  '
        f'RMSE = {rmse_ols:.2f} mN',
        fontsize=12, fontweight='bold')

    if out_dir:
        fig_path = out_dir / "CT_estimation.png"
        fig.savefig(fig_path, dpi=130)
        print(f"\n  Saved: {fig_path}")

        csv_path = out_dir / "CT_per_rpm.csv"
        with open(csv_path, 'w') as f:
            f.write("rpm_cmd,rpm_meas,T_mean_N,T_std_mN,n,"
                    "T_model_N,residual_mN\n")
            for row, Tp, res in zip(data_rows, T_pred_ols, resid_ols):
                f.write(f"{row[0]},{row[1]:.1f},{row[2]:.6f},"
                        f"{row[3]*1000:.4f},{row[4]},"
                        f"{Tp:.6f},{res*1000:.4f}\n")
        print(f"  Saved: {csv_path}")
        plt.close('all')
    else:
        plt.tight_layout()
        plt.show()

    print(f"\n{'='*58}")
    print(f"  FINAL RESULT")
    print(f"{'='*58}")
    print(f"  Tare (initial_weight) = {tare_gram:.4f} gram")
    print(f"  C_T (OLS)  = {C_T_ols:.6e} N/RPM²")
    print(f"  C_T (WLS)  = {C_T_wls:.6e} N/RPM²")
    print(f"  RMSE (OLS) = {rmse_ols:.2f} mN")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
