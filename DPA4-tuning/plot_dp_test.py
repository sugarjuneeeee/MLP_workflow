#!/usr/bin/env python3
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_txt(fname):
    """
    读取 DeePMD test 输出文件。
    自动跳过 # 开头的注释行。
    """
    if not os.path.exists(fname):
        print(f"[Skip] {fname} not found")
        return None

    try:
        data = np.loadtxt(fname, comments="#")
    except Exception as e:
        print(f"[Skip] failed to load {fname}: {e}")
        return None

    if data.size == 0:
        print(f"[Skip] {fname} is empty")
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    print(f"[Load] {fname}: shape = {data.shape}")
    return data


def calc_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    maxae = np.max(np.abs(err))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return mae, rmse, maxae, r2


def sample_xy(x, y, max_points=200000, seed=2024):
    """
    点太多时随机抽样，避免画图太慢。
    """
    n = len(x)
    if n <= max_points:
        return x, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def parity_plot(ax, y_true, y_pred, title, xlabel, ylabel, unit="", max_points=200000):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae, rmse, maxae, r2 = calc_metrics(y_true, y_pred)

    x_plot, y_plot = sample_xy(y_true, y_pred, max_points=max_points)

    ax.scatter(x_plot, y_plot, s=5, alpha=0.35, edgecolors="none")

    vmin = min(np.min(y_true), np.min(y_pred))
    vmax = max(np.max(y_true), np.max(y_pred))
    pad = 0.05 * (vmax - vmin + 1e-12)
    vmin -= pad
    vmax += pad

    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.2)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")

    txt = (
        f"N = {len(y_true)}\n"
        f"MAE = {mae:.4e} {unit}\n"
        f"RMSE = {rmse:.4e} {unit}\n"
        f"MaxAE = {maxae:.4e} {unit}\n"
        f"R$^2$ = {r2:.6f}"
    )

    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.grid(alpha=0.25)


def error_hist(ax, y_true, y_pred, title, xlabel, unit=""):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    err = y_pred[mask] - y_true[mask]

    mae, rmse, maxae, r2 = calc_metrics(y_true[mask], y_pred[mask])

    ax.hist(err, bins=80, alpha=0.8)
    ax.axvline(0.0, color="r", ls="--", lw=1.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)

    txt = (
        f"Mean err = {np.mean(err):.4e} {unit}\n"
        f"Std err = {np.std(err):.4e} {unit}\n"
        f"MAE = {mae:.4e} {unit}\n"
        f"RMSE = {rmse:.4e} {unit}"
    )

    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )


def get_energy(prefix):
    data = load_txt(f"{prefix}.e.out")
    if data is None or data.shape[1] < 2:
        return None
    return data[:, 0], data[:, 1]


def get_energy_per_atom(prefix):
    data = load_txt(f"{prefix}.e_peratom.out")
    if data is None or data.shape[1] < 2:
        return None
    return data[:, 0], data[:, 1]


def get_force(prefix):
    data = load_txt(f"{prefix}.f.out")
    if data is None or data.shape[1] < 6:
        return None

    # data_fx data_fy data_fz pred_fx pred_fy pred_fz
    y_true = data[:, 0:3].reshape(-1)
    y_pred = data[:, 3:6].reshape(-1)
    return y_true, y_pred


def get_tensor_9(prefix, suffix):
    """
    读取 stress 或 virial：
    data_xx ... data_zz pred_xx ... pred_zz
    一共 18 列。
    """
    data = load_txt(f"{prefix}.{suffix}.out")
    if data is None or data.shape[1] < 18:
        return None

    y_true = data[:, 0:9].reshape(-1)
    y_pred = data[:, 9:18].reshape(-1)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--prefix",
        default="test",
        help="dp test 输出前缀，例如 -d test 则 prefix=test"
    )
    parser.add_argument(
        "-o", "--output",
        default="test_summary.png",
        help="输出 png 文件名"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="scatter 最多绘制多少个点，力很多时建议限制"
    )
    args = parser.parse_args()

    prefix = args.prefix

    plots = []

    e = get_energy(prefix)
    if e is not None:
        plots.append({
            "name": "Total energy",
            "true": e[0],
            "pred": e[1],
            "unit": "eV",
            "xlabel": "DFT energy / eV",
            "ylabel": "DP energy / eV",
            "err_xlabel": "Pred - DFT energy / eV",
        })

    epa = get_energy_per_atom(prefix)
    if epa is not None:
        plots.append({
            "name": "Energy per atom",
            "true": epa[0],
            "pred": epa[1],
            "unit": "eV/atom",
            "xlabel": "DFT energy / eV/atom",
            "ylabel": "DP energy / eV/atom",
            "err_xlabel": "Pred - DFT energy / eV/atom",
        })

    f = get_force(prefix)
    if f is not None:
        plots.append({
            "name": "Force components",
            "true": f[0],
            "pred": f[1],
            "unit": "eV/Å",
            "xlabel": "DFT force / eV/Å",
            "ylabel": "DP force / eV/Å",
            "err_xlabel": "Pred - DFT force / eV/Å",
        })

    s = get_tensor_9(prefix, "s")
    if s is not None:
        plots.append({
            "name": "Stress components",
            "true": s[0],
            "pred": s[1],
            "unit": "eV/Å$^3$",
            "xlabel": "DFT stress / eV/Å$^3$",
            "ylabel": "DP stress / eV/Å$^3$",
            "err_xlabel": "Pred - DFT stress / eV/Å$^3$",
        })

    v = get_tensor_9(prefix, "v")
    if v is not None:
        plots.append({
            "name": "Virial components",
            "true": v[0],
            "pred": v[1],
            "unit": "eV",
            "xlabel": "DFT virial / eV",
            "ylabel": "DP virial / eV",
            "err_xlabel": "Pred - DFT virial / eV",
        })

    vp = get_tensor_9(prefix, "v_peratom")
    if vp is not None:
        plots.append({
            "name": "Virial per atom components",
            "true": vp[0],
            "pred": vp[1],
            "unit": "eV/atom",
            "xlabel": "DFT virial / eV/atom",
            "ylabel": "DP virial / eV/atom",
            "err_xlabel": "Pred - DFT virial / eV/atom",
        })

    if len(plots) == 0:
        raise RuntimeError("没有找到可画图的数据文件，请检查 test.*.out 是否存在。")

    nrows = len(plots)
    ncols = 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4.2 * nrows),
        constrained_layout=True
    )

    if nrows == 1:
        axes = np.array([axes])

    for i, item in enumerate(plots):
        y_true = item["true"]
        y_pred = item["pred"]

        parity_plot(
            axes[i, 0],
            y_true,
            y_pred,
            title=item["name"] + " parity",
            xlabel=item["xlabel"],
            ylabel=item["ylabel"],
            unit=item["unit"],
            max_points=args.max_points
        )

        error_hist(
            axes[i, 1],
            y_true,
            y_pred,
            title=item["name"] + " error distribution",
            xlabel=item["err_xlabel"],
            unit=item["unit"]
        )

    fig.suptitle(
        f"DeePMD test summary: prefix = {prefix}",
        fontsize=16,
        y=1.002
    )

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\n[Done] Saved figure to: {args.output}")

    print("\n========== Metrics summary ==========")
    for item in plots:
        mae, rmse, maxae, r2 = calc_metrics(item["true"], item["pred"])
        print(
            f"{item['name']:30s} "
            f"MAE = {mae:.6e} {item['unit']:10s}  "
            f"RMSE = {rmse:.6e} {item['unit']:10s}  "
            f"MaxAE = {maxae:.6e} {item['unit']:10s}  "
            f"R2 = {r2:.8f}"
        )


if __name__ == "__main__":
    main()
