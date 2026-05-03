#!/usr/bin/env python3
"""
C_T Estimation from Fixed-RPM Load Cell Measurements
=====================================================
T = C_T * omega^2  (no intercept, least-squares)

Directory structure:
    <data_dir>/
        initial_weight/   ← tare bag (motor OFF)
        1200RPM/
        2000RPM/
        ...
        7500RPM/

Usage:
    python3 compute_CT_without_intercept.py -d /path/to/Bag_file_with_prop_gauard
    python3 compute_CT_without_intercept.py -d /path/to/bags --no_save
    python3 compute_CT_without_intercept.py -d /path/to/bags -o ./results
"""

# ── Auto-install ────────────────────────────────────────────────────────
import subprocess, sys
for _p in ["rosbags", "numpy", "matplotlib"]:
    try: __import__(_p.replace("-","_"))
    except ImportError:
        print(f"[setup] pip install {_p} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _p,
                               "--break-system-packages", "-q"])
# ───────────────────────────────────────────────────────────────────────

import argparse, struct, sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

G         = 9.81   # m/s²
MAD_SIGMA = 5.0       # spike rejection threshold

typestore = get_typestore(Stores.ROS2_HUMBLE)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_rpm(data: bytes) -> int:
    return struct.unpack_from('<i', data, 20)[0]


def mad_clean(arr: np.ndarray, sigma: float = MAD_SIGMA) -> np.ndarray:
    """MAD 기반 spike 제거 후 clean array 반환."""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-9: mad = 1e-9
    return arr[np.abs(arr - med) / (1.4826 * mad) < sigma]


def load_bag(ds_path: str):
    """
    bag 폴더 → (load_cell_gram_clean, rpm_array)
    load cell 단위: gram (raw, tare 미적용)
    """
    p   = Path(ds_path)
    db  = list(p.glob("*.db3"))[0]
    conn = sqlite3.connect(str(db)); cur = conn.cursor()
    cur.execute("SELECT id,name FROM topics")
    tm  = {n: i for i, n in cur.fetchall()}; conn.close()

    # Load cell [gram]
    thr_raw = []
    with Reader(str(p)) as reader:
        tc = [c for c in reader.connections if c.topic == '/load_cell/weight']
        if not tc: return None, None
        for c, ts, raw in reader.messages(connections=tc):
            msg = typestore.deserialize_cdr(raw, c.msgtype)
            thr_raw.append(msg.point.z)
    thr_clean = mad_clean(np.array(thr_raw))

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
        description="C_T = T / ω²  estimation from fixed-RPM load cell bags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--data_dir", "-d", required=True,
                        help="initial_weight/ 와 <N>RPM/ 폴더가 있는 디렉토리")
    parser.add_argument("--out_dir",  "-o", default=None,
                        help="결과 저장 폴더 (기본: <data_dir>/ct_results)")
    parser.add_argument("--no_save",  action="store_true",
                        help="저장 없이 plt.show() 표시")
    parser.add_argument("--mass_kg",  type=float, nargs="+",
                        default=[2.960, 3.230],
                        help="호버 RPM 역산용 기체 질량 [kg] (기본: 2.960 3.230)")
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

    print(f"\n{'='*56}")
    print(f"  C_T Estimation   T = C_T · ω²  (no intercept)")
    print(f"{'='*56}")
    print(f"  data_dir : {data_dir}")
    print(f"  output   : {'display only' if args.no_save else out_dir}\n")

    # ── [1] Tare ──────────────────────────────────────────────────────
    tare_path = data_dir / "initial_weight"
    if not tare_path.exists():
        print("[ERROR] initial_weight/ 폴더 없음"); sys.exit(1)

    tare_raw, _ = load_bag(str(tare_path))
    tare_gram   = tare_raw.mean()

    print(f"[1/3] Tare (initial_weight, motor OFF)")
    print(f"  mean = {tare_gram:.4f} gram  |  "
          f"std = {tare_raw.std():.4f} g  |  n = {len(tare_raw)}\n")

    # ── [2] Per-RPM 로드 ──────────────────────────────────────────────
    rpm_dirs = sorted(
        [d for d in data_dir.iterdir()
         if d.is_dir() and d.name != 'initial_weight' and 'RPM' in d.name.upper()],
        key=lambda x: int(''.join(filter(str.isdigit, x.name)))
    )
    if not rpm_dirs:
        print("[ERROR] RPM bag 폴더 없음"); sys.exit(1)

    print(f"[2/3] Loading {len(rpm_dirs)} RPM bags ...")
    print(f"  {'RPM_cmd':>8} {'RPM_meas':>9} {'T_mean [N]':>11} "
          f"{'T_std [mN]':>11} {'C_T_i [N/RPM²]':>16} {'n':>6}")
    print(f"  {'-'*68}")

    rpm_cmd_list, rpm_meas_list = [], []
    T_mean_list, T_std_list     = [], []
    n_list                      = []

    for ds in rpm_dirs:
        thr_clean, rpm = load_bag(str(ds))
        if thr_clean is None: continue

        rpm_cmd  = int(''.join(filter(str.isdigit, ds.name)))
        # T [N] = (measured_gram - tare_gram) * 1e-3 * G
        T_N      = (thr_clean - tare_gram) * 1e-3 * G
        rpm_meas = float(rpm.mean()) if len(rpm) else float(rpm_cmd)

        # 개별 C_T
        C_T_i = T_N.mean() / (rpm_meas ** 2)

        rpm_cmd_list.append(rpm_cmd)
        rpm_meas_list.append(rpm_meas)
        T_mean_list.append(T_N.mean())
        T_std_list.append(T_N.std())
        n_list.append(len(T_N))

        print(f"  {rpm_cmd:8d} {rpm_meas:9.1f} {T_N.mean():11.4f} "
              f"{T_N.std()*1000:11.2f} {C_T_i:.6e} {len(T_N):6d}")

    rpm_cmd_arr  = np.array(rpm_cmd_list,  dtype=float)
    rpm_meas_arr = np.array(rpm_meas_list, dtype=float)
    T_arr        = np.array(T_mean_list)
    T_std_arr    = np.array(T_std_list)
    CT_i_arr     = T_arr / rpm_meas_arr**2   # per-point C_T

    # ── [3] C_T 추정 (no intercept) ──────────────────────────────────
    print(f"\n[3/3] C_T estimation  (T = C_T · ω², no intercept)")

    w2 = rpm_meas_arr ** 2

    # OLS: min Σ(T - C_T·ω²)²  →  C_T = <ω², T> / <ω², ω²>
    C_T_ols = np.dot(w2, T_arr) / np.dot(w2, w2)

    # WLS: min Σ (T - C_T·ω²)² / σ²
    weights = 1.0 / np.where(T_std_arr > 0, T_std_arr**2, 1e-18)
    C_T_wls = np.dot(weights * w2, T_arr) / np.dot(weights * w2, w2)

    # Mean of per-point C_T (simple mean & weighted mean)
    C_T_mean     = CT_i_arr.mean()
    C_T_mean_wls = np.average(CT_i_arr, weights=weights)

    T_pred = C_T_ols * w2
    resid  = T_arr - T_pred
    rmse   = np.sqrt(np.mean(resid**2)) * 1000
    r2     = 1 - np.sum(resid**2) / np.sum((T_arr - T_arr.mean())**2)

    print(f"\n  {'Method':<35} {'C_T [N/RPM²]':>16}")
    print(f"  {'-'*54}")
    print(f"  {'OLS  Σ(T-C_T·ω²)²':<35} {C_T_ols:.6e}  ← recommended")
    print(f"  {'WLS  Σ(T-C_T·ω²)²/σ²':<35} {C_T_wls:.6e}")
    print(f"  {'Mean of C_T_i = T_i/ω_i²':<35} {C_T_mean:.6e}")
    print(f"  {'Weighted mean of C_T_i':<35} {C_T_mean_wls:.6e}")
    print(f"\n  RMSE (OLS) = {rmse:.2f} mN  |  R² = {r2:.6f}")

    # Hover RPM
    print(f"\n  Hover RPM (6 rotors, C_T_OLS):")
    for m in args.mass_kg:
        T_h   = m * G / 6
        rpm_h = np.sqrt(T_h / C_T_ols)
        print(f"    m = {m:.3f} kg  →  T/rotor = {T_h:.4f} N  "
              f"→  RPM_hover = {rpm_h:.1f}")

    # ── Plots ─────────────────────────────────────────────────────────
    rpm_fit  = np.linspace(0, rpm_meas_arr.max() * 1.05, 500)
    T_fit    = C_T_ols * rpm_fit**2

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # [0,0] RPM vs T + fit
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(rpm_meas_arr, T_arr, yerr=T_std_arr * 3,
                fmt='o', color='#1565C0', ms=8, lw=1.5, capsize=5,
                label='Measured (mean ± 3σ)', zorder=5)
    ax.plot(rpm_fit, T_fit, 'r-', lw=2.5,
            label=f'C_T · ω²  (C_T={C_T_ols:.4e})')
    ax.set_xlabel('RPM', fontsize=12)
    ax.set_ylabel('Thrust [N]', fontsize=12)
    ax.set_title('RPM vs Thrust', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # [0,1] Linearity: T_pred vs T_meas
    ax = fig.add_subplot(gs[0, 1])
    T_pred_pts = C_T_ols * rpm_meas_arr**2
    ax.scatter(T_pred_pts, T_arr, s=70, color='steelblue',
               edgecolors='navy', zorder=5)
    for xp, yp, rpm in zip(T_pred_pts, T_arr, rpm_cmd_arr):
        ax.annotate(f'{int(rpm)}', (xp, yp),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=8, color='gray')
    lim = [0, max(T_arr) * 1.05]
    ax.plot(lim, lim, 'r--', lw=2, label='1:1 line')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('C_T · ω² [N]', fontsize=12)
    ax.set_ylabel('T measured [N]', fontsize=12)
    ax.set_title('Linearity: Model vs Measured', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # [1,0] Residuals
    ax = fig.add_subplot(gs[1, 0])
    colors = ['#E53935' if r < 0 else '#43A047' for r in resid]
    bars   = ax.bar(rpm_cmd_arr, resid * 1000, width=200,
                    color=colors, edgecolor='gray')
    for bar, v in zip(bars, resid * 1000):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (1 if v >= 0 else -3),
                f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(0, color='k', lw=1.5)
    ax.set_xlabel('RPM', fontsize=12)
    ax.set_ylabel('T_meas − C_T·ω² [mN]', fontsize=12)
    ax.set_title(f'Residuals  (RMSE = {rmse:.1f} mN)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # [1,1] Per-point C_T
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rpm_cmd_arr, CT_i_arr * 1e7, 'o-', color='#1565C0',
            ms=8, lw=1.8, label='C_T_i = T_i / ω_i²')
    ax.axhline(C_T_ols * 1e7, color='red',   lw=2, ls='--',
               label=f'OLS  = {C_T_ols:.4e}')
    ax.axhline(C_T_wls * 1e7, color='green', lw=1.8, ls=':',
               label=f'WLS  = {C_T_wls:.4e}')
    ax.set_xlabel('RPM', fontsize=12)
    ax.set_ylabel('C_T × 10⁷  [N/RPM²]', fontsize=12)
    ax.set_title('Per-point C_T = T / ω²', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'C_T Estimation  —  With Prop Guard\n'
        f'★  C_T (OLS) = {C_T_ols:.6e} N/RPM²'
        f'   |   RMSE = {rmse:.2f} mN   |   R² = {r2:.6f}',
        fontsize=12, fontweight='bold')

    if out_dir:
        fp = out_dir / "CT_estimation.png"
        fig.savefig(fp, dpi=600)
        print(f"\n  Saved: {fp}")

        cp = out_dir / "CT_per_rpm.csv"
        with open(cp, 'w') as f:
            f.write("rpm_cmd,rpm_meas,T_mean_N,T_std_mN,"
                    "C_T_i,T_model_N,residual_mN,n\n")
            for i in range(len(rpm_cmd_arr)):
                f.write(f"{int(rpm_cmd_arr[i])},{rpm_meas_arr[i]:.1f},"
                        f"{T_arr[i]:.6f},{T_std_arr[i]*1000:.4f},"
                        f"{CT_i_arr[i]:.6e},{T_pred[i]:.6f},"
                        f"{resid[i]*1000:.4f},{n_list[i]}\n")
        print(f"  Saved: {cp}")
        plt.close('all')
    else:
        plt.tight_layout()
        plt.show()

    print(f"\n{'='*56}")
    print(f"  FINAL RESULT")
    print(f"{'='*56}")
    print(f"  Tare          = {tare_gram:.4f} gram")
    print(f"  C_T (OLS)     = {C_T_ols:.6e} N/RPM²")
    print(f"  C_T (WLS)     = {C_T_wls:.6e} N/RPM²")
    print(f"  RMSE          = {rmse:.2f} mN")
    print(f"  R²            = {r2:.6f}")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()