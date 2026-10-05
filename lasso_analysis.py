"""题目一：酱油风味物质 Lasso 降维。

运行方式：在本文件所在目录执行 ``python lasso_analysis.py``。
输入为 data/soy_sauce_flavor_standardized.csv，输出保存到 outputs/。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script also runs on headless machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, lasso_path


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "soy_sauce_flavor_standardized.csv"
OUT_DIR = ROOT
FIG_DIR = OUT_DIR / "figures"

NAME_MAP = {
    "丙酮酸": "Pyruvic acid", "酒石酸": "Tartaric acid", "L-苹果酸": "Malic acid",
    "焦谷氨酸": "Pyroglutamic acid", "乳酸": "Lactic acid", "富马酸": "Fumaric acid",
    "琥珀酸": "Succinic acid", "草酸": "Oxalic acid", "柠檬酸": "Citric acid",
    "维生素C": "Ascorbic acid", "5'-AMP": "5'-AMP", "5'-GMP": "5'-GMP",
    "5'-CMP": "5'-CMP", "5'-UMP": "5'-UMP", "5'-IMP": "5'-IMP",
    "Ala丙氨酸": "Ala", "Arg精氨酸": "Arg", "Asp天冬氨酸": "Asp",
    "Glu谷氨酸": "Glu", "Gly甘氨酸": "Gly", "His组氨酸": "His",
    "Ile异亮氨酸": "Ile", "Leu亮氨酸": "Leu", "Lys赖氨酸": "Lys",
    "Met蛋氨酸": "Met", "Phe苯丙氨酸": "Phe", "Pro脯氨酸": "Pro",
    "Ser丝氨酸": "Ser", "Thr苏氨酸": "Thr", "Trp色氨酸": "Trp",
    "Tyr酪氨酸": "Tyr", "Val缬氨酸": "Val", "cys胱氨酸": "Cys",
    "Na+": "Na+", "K+": "K+", "Ca2+": "Ca2+", "Mg2+": "Mg2+",
    "Cl-": "Cl-", "NSSS": "NSSS", "NaCl": "NaCl",
}


def make_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def choose_alpha_1se(alphas: np.ndarray, mse_path: np.ndarray) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Return the largest alpha within one standard error of the minimum."""
    mean_mse = mse_path.mean(axis=1)
    se_mse = mse_path.std(axis=1, ddof=1) / np.sqrt(mse_path.shape[1])
    min_idx = int(np.argmin(mean_mse))
    cutoff = mean_mse[min_idx] + se_mse[min_idx]
    eligible = np.flatnonzero(mean_mse <= cutoff)
    # alphas_ is descending, so the first eligible entry is the largest alpha.
    idx_1se = int(eligible[0])
    return float(alphas[idx_1se]), idx_1se, mean_mse, se_mse


def configure_plot() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })


def main() -> None:
    make_dirs()
    configure_plot()

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    X = df.drop(columns=["ID", "咸味评分"])
    y = df["咸味评分"]
    feature_names = X.columns.to_list()
    feature_labels = [NAME_MAP.get(name, name) for name in feature_names]

    try:
        model = LassoCV(
            alphas=200, cv=10, random_state=42, max_iter=20000,
            fit_intercept=True, n_jobs=None,
        )
    except TypeError:  # compatibility with older scikit-learn releases
        model = LassoCV(
            n_alphas=200, cv=10, random_state=42, max_iter=20000,
            fit_intercept=True, n_jobs=None,
        )
    model.fit(X, y)

    alpha_min = float(model.alpha_)
    alpha_1se, idx_1se, mean_mse, se_mse = choose_alpha_1se(model.alphas_, model.mse_path_)
    idx_min = int(np.argmin(mean_mse))
    coef_min = model.coef_.copy()
    nonzero = np.abs(coef_min) > 1e-10
    selected = pd.DataFrame({"feature": feature_names, "label": feature_labels, "coefficient": coef_min})
    selected = selected.loc[nonzero].sort_values("coefficient")
    zero_features = [feature_labels[i] for i, keep in enumerate(nonzero) if not keep]

    # Coefficient path across the same alpha grid used by LassoCV.
    _, path_coefs, _ = lasso_path(X.to_numpy(), y.to_numpy(), alphas=model.alphas_, max_iter=20000)
    log_alphas = np.log10(model.alphas_)
    line_colors = plt.cm.tab20(np.linspace(0, 1, len(feature_names)))

    # Figure 1: coefficient profiles.
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for j, label in enumerate(feature_labels):
        ax.plot(log_alphas, path_coefs[j], color=line_colors[j], lw=0.85, alpha=0.78, label=label)
    ax.axvline(np.log10(alpha_min), color="black", ls="--", lw=1.3,
               label=fr"$log_{{10}}(\lambda_{{min}})$ = {np.log10(alpha_min):.2f}")
    ax.axvline(np.log10(alpha_1se), color="firebrick", ls=":", lw=1.5,
               label=fr"$log_{{10}}(\lambda_{{1se}})$ = {np.log10(alpha_1se):.2f}")
    ax.set_title("Lasso Coefficient Profiles")
    ax.set_xlabel(r"$log_{10}(\lambda)$")
    ax.set_ylabel("Coefficient")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_lasso_paths.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: ten-fold CV MSE and standard error.
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    ax.errorbar(log_alphas, mean_mse, yerr=se_mse, fmt="o-", ms=2.8, lw=0.9,
                elinewidth=0.55, capsize=1.7, color="navy", alpha=0.82)
    ax.axvline(np.log10(alpha_min), color="black", ls="--", lw=1.3,
               label=fr"$lambda_{{min}}$ = {alpha_min:.6g} (log = {np.log10(alpha_min):.2f})")
    ax.axvline(np.log10(alpha_1se), color="firebrick", ls=":", lw=1.5,
               label=fr"$lambda_{{1se}}$ = {alpha_1se:.6g} (log = {np.log10(alpha_1se):.2f})")
    ax.scatter([np.log10(alpha_min)], [mean_mse[idx_min]], color="black", s=24, zorder=5)
    ax.set_title("10-Fold Cross-Validation Error")
    ax.set_xlabel(r"$log_{10}(\lambda)$")
    ax.set_ylabel("Cross-validation MSE")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cv_mse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: non-zero coefficients at lambda_min.
    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.29 * len(selected) + 1.2)))
    bars = ax.barh(selected["label"], selected["coefficient"], color="#4472C4", edgecolor="white")
    max_abs = max(float(np.max(np.abs(selected["coefficient"]))), 1e-9)
    for bar, value in zip(bars, selected["coefficient"]):
        offset = 0.012 * max_abs if value >= 0 else -0.012 * max_abs
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(r"Non-zero Lasso Coefficients at $\lambda_{min}$")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Flavor indicator")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save machine-readable results for auditing and use in the Markdown report.
    result = {
        "n_rows": int(df.shape[0]), "n_features": int(X.shape[1]), "cv_folds": 10,
        "alpha_min": alpha_min, "log10_alpha_min": float(np.log10(alpha_min)),
        "alpha_1se": alpha_1se, "log10_alpha_1se": float(np.log10(alpha_1se)),
        "min_cv_mse": float(mean_mse[idx_min]), "min_cv_se": float(se_mse[idx_min]),
        "n_nonzero_at_min": int(nonzero.sum()), "zero_features": zero_features,
    }
    pd.DataFrame([result]).to_json(OUT_DIR / "lasso_summary.json", orient="records", force_ascii=False, indent=2)
    selected.to_csv(OUT_DIR / "selected_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "alpha": model.alphas_, "log10_alpha": log_alphas,
        "mean_mse": mean_mse, "se_mse": se_mse,
    }).to_csv(OUT_DIR / "cv_path.csv", index=False, encoding="utf-8-sig")
    print(pd.Series(result).to_string())


if __name__ == "__main__":
    main()
