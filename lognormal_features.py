import numpy as np
from scipy.stats import multivariate_normal, ks_2samp
from scipy.spatial.distance import jensenshannon

def extract_lognormal_cov_features(mu, sigma, cov_mat, log_energy_windows, eps=1e-12):
    """
    Extract compact, interpretable features from
    multivariate log-normal EEG energy model.

    Parameters
    ----------
    mu : ndarray, shape (n_channels,)
        Mean of log-energy per channel
    sigma : ndarray, shape (n_channels,)
        Std of log-energy per channel
    cov_mat : ndarray, shape (n_channels, n_channels)
        Covariance of log-energy across channels
    log_energy_windows : ndarray, shape (n_windows, n_channels), optional
        Window-wise log-energy samples (required for likelihood and distribution metrics)

    Returns
    -------
    features : dict
        Dictionary of scalar features
    """

    n = cov_mat.shape[0]

    # -----------------------------
    # Covariance-based features
    # -----------------------------
    diag = np.diag(cov_mat)
    off_diag = cov_mat - np.diag(diag)

    #auto_var_mean = np.mean(diag)
    cross_cov_mean = np.sum(np.abs(off_diag)) / (n * (n - 1))
    #cross_cov_mean_signed = np.sum(off_diag) / (n * (n - 1))
    trace_cov = np.trace(cov_mat)

    eigvals = np.linalg.eigvalsh(cov_mat)
    lambda_max = eigvals[-1]

    p = eigvals / (np.sum(eigvals) + eps)
    eig_entropy = -np.sum(p * np.log(p + eps))

    # -----------------------------
    # μ-based features
    # -----------------------------
    mu_mean = np.mean(mu)
    mu_std = np.std(mu)

    # -----------------------------
    # σ-based features
    # -----------------------------
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)

    # ============================================================
    # NEW FEATURES (ADDITIONS ONLY)
    # ============================================================

    # -----------------------------
    # 1. Normalized log-likelihood
    # -----------------------------
    # Measures how well all windowed log-energy vectors fit the
    # estimated multivariate Gaussian model.

    if log_energy_windows is not None:

        # -------------------------------------------------
        # Regularize covariance ONLY for likelihood
        # (does NOT affect covariance-based features)
        # -------------------------------------------------
        reg_strength = 1e-6 * np.trace(cov_mat) / n
        cov_ll = cov_mat + reg_strength * np.eye(n)

        mvn = multivariate_normal(
            mean=mu,
            cov=cov_ll,
            allow_singular=False
        )

        log_likelihood = mvn.logpdf(log_energy_windows)

        # Mean log-likelihood across windows
        ll_mean = np.mean(log_likelihood)
        ll_std = np.std(log_likelihood)
        ll_p05 = np.percentile(log_likelihood, 5)

        # Safety clip to avoid numerical explosions
        # log_likelihood_mean = np.clip(log_likelihood_mean, -1e6, 1e6)

    else:
        log_likelihood_mean = np.nan

    # import matplotlib.pyplot as plt
    # plt.close()
    # plt.hist(log_likelihood, bins=40)
    # plt.xlabel("Log-likelihood")
    # plt.ylabel("Count")
    # plt.title("Distribution of log-likelihood across windows")
    # plt.savefig("loglikelihood_hist.png", dpi=200, bbox_inches='tight')
    # plt.close()

    # NOTE:
    # Min–max normalization must be done ACROSS SUBJECTS.
    # Here we return the raw mean log-likelihood.
    # Normalization should be applied later once all subjects are processed.

    # -----------------------------
    # 2. (1 − KSPR) × Jensen–Shannon Divergence
    # -----------------------------
    # Measures similarity of channel-wise distributions.
    distribution_similarity = np.nan
    combined_similarity = np.nan
    S_KS = np.nan
    S_JSD = np.nan

    if log_energy_windows is not None:

        js_divs = []
        ks_sims = []
        js_sims = []

        for i in range(n):
            for j in range(i + 1, n):

                # ---------- KS ----------
                D, _ = ks_2samp(
                    log_energy_windows[:, i],
                    log_energy_windows[:, j]
                )
                #ks_pvals.append(pval)
                ks_sims.append(1.0 - D)  # similarity

                # ---------- JS ----------
                hist_i, _ = np.histogram(log_energy_windows[:, i], bins=50, density=True)
                hist_j, _ = np.histogram(log_energy_windows[:, j], bins=50, density=True)

                hist_i = hist_i.astype(float) + eps
                hist_j = hist_j.astype(float) + eps

                hist_i /= np.sum(hist_i)
                hist_j /= np.sum(hist_j)

                js = jensenshannon(hist_i, hist_j)

                js_divs.append(js)  # divergence
                js_sims.append(1.0 - js)  # similarity

        # ---- Old metric ----
        S_KS = np.mean(ks_sims)
        S_JSD = np.mean(js_sims)
        distribution_similarity = S_KS * S_JSD

        # ---- New combined metric ----
        w = 0.5
        combined_similarity = w * S_JSD + (1 - w) * S_KS  # 75% JSD, 25% KS  (your current choice)

    # -----------------------------
    # 3. Stability metric (diagonal rigidity) & 4. Coupling Factor
    # -----------------------------
    # Coupling: Fraction of total covariance energy coming from off-diagonal terms.

    # ----------------------------
    # Separate diagonal and off-diagonal
    # ----------------------------
    diag = np.diag(cov_mat)                   # variances (>= 0 in theory)
    off_diag = cov_mat - np.diag(diag)        # pure covariances

    off_mask = ~np.eye(n, dtype=bool)         # Mask for off-diagonal entries

    # ----------------------------
    # Diagonal statistics (self-energy)
    # ----------------------------
    diag_mean = np.mean(diag)                    # Mean variance across channels (self-energy)
    diag_std = np.std(diag) + eps

    # Stability: bounded, smooth measure of variance homogeneity
    # High when variances are similar across channels
    stability = diag_mean / (diag_mean + diag_std)

    # Efficiency: penalizes variance dispersion
    # Can be < 0 if dispersion dominates (allowed, informative)
    efficiency = 1.0 - (diag_std / (diag_mean + eps))

    # ----------------------------
    # Coupling metrics (interaction strength)
    # ----------------------------
    offdiag_mean = np.mean(np.abs(off_diag[off_mask]))
    offdiag_sum = np.sum(np.abs(off_diag))

    diag_sum = np.sum(diag)

    # Mean-based coupling (PRIMARY, N-invariant)
    coupling_mean = offdiag_mean / (diag_mean + offdiag_mean + eps)

    # Sum-based coupling (SECONDARY, interaction mass)
    coupling_sum = offdiag_sum / (diag_sum + offdiag_sum + eps)
    return {
        "cross_cov_mean": cross_cov_mean,
        "trace_cov": trace_cov,
        "lambda_max": lambda_max,
        "eig_entropy": eig_entropy,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,

        # New features for log-likelihood and distribution similarity
        "ll_mean": ll_mean,
        "ll_std": ll_std,
        "ll_p05": ll_p05,
        "distribution_similarity": distribution_similarity,
        "combined_similarity": combined_similarity,
        "S_KS": S_KS,
        "S_JSD": S_JSD,

        # Stability and Coupling features
        # Self-energy descriptors
        #"diag_mean": diag_mean,
        "diag_std": diag_std,

        # Network organization
        "stability": stability,
        "efficiency": efficiency,

        # Coupling
        "coupling_mean": coupling_mean,
        "coupling_sum": coupling_sum,

        # Raw interaction magnitudes (optional diagnostics)
        "offdiag_mean": offdiag_mean,
        "offdiag_sum": offdiag_sum,
        "diag_sum": diag_sum,
    }
