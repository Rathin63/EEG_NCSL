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

    auto_var_mean = np.mean(diag)
    cross_cov_mean = np.sum(np.abs(off_diag)) / (n * (n - 1))
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
        log_likelihood_mean = np.mean(log_likelihood)

        # Safety clip to avoid numerical explosions
        log_likelihood_mean = np.clip(log_likelihood_mean, -1e6, 1e6)

    else:
        log_likelihood_mean = np.nan

    # NOTE:
    # Min–max normalization must be done ACROSS SUBJECTS.
    # Here we return the raw mean log-likelihood.
    # Normalization should be applied later once all subjects are processed.

    # -----------------------------
    # 2. (1 − KSPR) × Jensen–Shannon Divergence
    # -----------------------------
    # Measures similarity of channel-wise distributions.

    if log_energy_windows is not None:
        ks_pvals = []
        js_divs = []

        for i in range(n):
            for j in range(i + 1, n):
                # KS test between channel distributions
                _, pval = ks_2samp(
                    log_energy_windows[:, i],
                    log_energy_windows[:, j]
                )
                ks_pvals.append(pval)

                # Jensen–Shannon divergence (histogram-based)
                hist_i, _ = np.histogram(log_energy_windows[:, i], bins=50, density=True)
                hist_j, _ = np.histogram(log_energy_windows[:, j], bins=50, density=True)

                hist_i += eps
                hist_j += eps

                js = jensenshannon(hist_i, hist_j)
                js_divs.append(js)

        kspr_mean = np.mean(ks_pvals)
        jd_mean = np.mean(js_divs)

        distribution_similarity = (1.0 - kspr_mean) * jd_mean
    else:
        distribution_similarity = np.nan

    # -----------------------------
    # 3. Coupling Factor
    # -----------------------------
    # Fraction of total covariance energy coming from off-diagonal terms.

    total_cov_sum = np.sum(np.abs(cov_mat)) + eps
    off_diag_sum = np.sum(np.abs(off_diag))
    coupling_factor = off_diag_sum / total_cov_sum

    # -----------------------------
    # 4. Stability metric (diagonal rigidity)
    # -----------------------------
    # High when diagonal variances are strong and homogeneous.

    diag_mean = np.mean(diag)
    diag_std = np.std(diag) + eps
    stability = 1.0 - (1.0 / (1.0 + diag_mean / diag_std))

    return {
        "auto_var_mean": auto_var_mean,
        "cross_cov_mean": cross_cov_mean,
        "trace_cov": trace_cov,
        "lambda_max": lambda_max,
        "eig_entropy": eig_entropy,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,

        # New features
        "log_likelihood_mean": log_likelihood_mean,
        "distribution_similarity": distribution_similarity,
        "coupling_factor": coupling_factor,
        "stability": stability,
    }
