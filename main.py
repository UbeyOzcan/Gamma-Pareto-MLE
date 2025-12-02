import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma, digamma, polygamma, gammainc
from scipy.optimize import minimize, fsolve
from scipy.stats import gamma as gamma_dist
import warnings

warnings.filterwarnings('ignore')


class GammaParetoMLE:
    """
    Maximum Likelihood Estimator for Gamma-Pareto Distribution

    Based on: Alzaatreh, Famoye & Lee (2012)
    Gamma-Pareto Distribution and Its Applications

    The Gamma-Pareto PDF is given by:
    g(x) = (1/(x*Γ(α)*c^α)) * (θ/x)^(1/c) * [log(x/θ)]^(α-1)

    for x > θ, α > 0, c > 0, θ > 0
    """

    def __init__(self, data=None):
        """
        Initialize the Gamma-Pareto MLE estimator

        Parameters:
        -----------
        data : array-like, optional
            Data to fit
        """
        self.data = None
        if data is not None:
            self.data = np.array(data)

        # Estimated parameters (using paper's notation)
        self.alpha_hat = None  # α
        self.c_hat = None  # c = β/k in paper
        self.theta_hat = None  # θ (minimum value)

        # Standard errors
        self.alpha_se = None
        self.c_se = None
        self.theta_se = None

        # Log-likelihood
        self.log_likelihood = None

        # Sample statistics
        self.x_min = None  # Minimum of data (used for theta)
        self.n_min_freq = None  # Frequency of minimum value

    def pdf(self, x, alpha, c, theta):
        """
        Probability density function of Gamma-Pareto distribution

        From equation (2.2) in the paper:
        g(x) = 1/(x*Γ(α)*c^α) * (θ/x)^(1/c) * [log(x/θ)]^(α-1)

        Parameters:
        -----------
        x : array-like
            Values at which to evaluate PDF
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns:
        --------
        pdf values
        """
        x = np.array(x)
        mask = x > theta

        result = np.zeros_like(x, dtype=float)

        if np.any(mask):
            x_masked = x[mask]
            term1 = 1 / (x_masked * gamma(alpha) * c ** alpha)
            term2 = (theta / x_masked) ** (1 / c)
            term3 = np.log(x_masked / theta) ** (alpha - 1)
            result[mask] = term1 * term2 * term3

        return result

    def cdf(self, x, alpha, c, theta):
        """
        Cumulative distribution function of Gamma-Pareto distribution

        From equation (2.3) in the paper:
        G(x) = γ(α, (1/c)*log(x/θ)) / Γ(α)

        where γ(α, t) is the lower incomplete gamma function

        Parameters:
        -----------
        x : array-like
            Values at which to evaluate CDF
        alpha, c, theta : float
            Distribution parameters

        Returns:
        --------
        cdf values
        """
        x = np.array(x)
        mask = x > theta

        result = np.zeros_like(x, dtype=float)
        result[x <= theta] = 0

        if np.any(mask):
            x_masked = x[mask]
            # Using gammainc which is the regularized incomplete gamma function
            # gammainc(a, x) = γ(a, x) / Γ(a)
            result[mask] = gammainc(alpha, (1 / c) * np.log(x_masked / theta))

        return result

    def hazard_function(self, x, alpha, c, theta):
        """
        Hazard function of Gamma-Pareto distribution

        From equation (3.1) in the paper:
        h(x) = g(x) / (1 - G(x))

        Parameters:
        -----------
        x : array-like
            Values at which to evaluate hazard function
        alpha, c, theta : float
            Distribution parameters

        Returns:
        --------
        hazard function values
        """
        x = np.array(x)
        mask = x > theta

        result = np.zeros_like(x, dtype=float)

        if np.any(mask):
            x_masked = x[mask]
            pdf_vals = self.pdf(x_masked, alpha, c, theta)
            cdf_vals = self.cdf(x_masked, alpha, c, theta)
            result[mask] = pdf_vals / (1 - cdf_vals)

        return result

    def moments(self, r, alpha, c, theta):
        """
        r-th non-central moment of Gamma-Pareto distribution

        From equation (4.2) in the paper:
        E(X^r) = θ^r * (1 - r*c)^(-α), for c < 1/r

        Parameters:
        -----------
        r : int
            Order of moment
        alpha, c, theta : float
            Distribution parameters

        Returns:
        --------
        r-th moment
        """
        if r * c >= 1:
            return np.inf
        return theta ** r * (1 - r * c) ** (-alpha)

    def mean(self, alpha, c, theta):
        """Mean of Gamma-Pareto distribution"""
        return self.moments(1, alpha, c, theta)

    def variance(self, alpha, c, theta):
        """Variance of Gamma-Pareto distribution"""
        if 2 * c >= 1:
            return np.inf
        moment1 = self.moments(1, alpha, c, theta)
        moment2 = self.moments(2, alpha, c, theta)
        return moment2 - moment1 ** 2

    def mode(self, alpha, c, theta):
        """
        Mode of Gamma-Pareto distribution

        From Theorem 2 in the paper:
        - If α ≤ 1: mode = θ
        - If α > 1: mode = θ * exp(c*(α-1)/(c+1))
        """
        if alpha <= 1:
            return theta
        else:
            return theta * np.exp(c * (alpha - 1) / (c + 1))

    def _log_likelihood_function(self, params, data, theta=None):
        """
        Log-likelihood function for Gamma-Pareto distribution

        From equation (5.1) in the paper:
        log L(α, c) = Σ [ -α log c - log Γ(α) - log θ - (1 + 1/c)*log(x_i/θ) 
                        + (α - 1)*log(log(x_i/θ)) ]

        Following Smith's method (θ is estimated by sample minimum)
        """
        if theta is None:
            theta = np.min(data)

        alpha, c = params

        # Check parameter constraints
        if alpha <= 0 or c <= 0:
            return -np.inf

        # Exclude minimum values (following Smith's method)
        mask = data > theta
        data_filtered = data[mask]

        if len(data_filtered) == 0:
            return -np.inf

        n = len(data_filtered)

        # Calculate terms
        log_ratio = np.log(data_filtered / theta)
        log_log_ratio = np.log(log_ratio)

        # Log-likelihood components
        term1 = -alpha * np.log(c)
        term2 = -np.log(gamma(alpha))
        term3 = -np.log(theta)  # Note: -log(θ) not -log(x_{(1)}) in paper
        term4 = -(1 + 1 / c) * log_ratio
        term5 = (alpha - 1) * log_log_ratio

        log_lik = np.sum(term1 + term2 + term3 + term4 + term5)

        return log_lik

    def _score_equations(self, params, data, theta):
        """
        Score equations (first derivatives of log-likelihood)

        From equations (5.2) and (5.3) in the paper
        """
        alpha, c = params

        # Exclude minimum values
        mask = data > theta
        data_filtered = data[mask]

        if len(data_filtered) == 0:
            return np.array([0, 0])

        n = len(data_filtered)
        log_ratio = np.log(data_filtered / theta)
        log_log_ratio = np.log(log_ratio)

        # Partial derivatives
        # Equation (5.2): ∂logL/∂α
        dL_dalpha = np.sum(-np.log(c) - digamma(alpha) + log_log_ratio)

        # Equation (5.3): ∂logL/∂c
        dL_dc = np.sum(-alpha / c + (1 / c ** 2) * log_ratio)

        return np.array([dL_dalpha, dL_dc])

    def fit(self, data, initial_params=None, method='hybrid'):
        """
        Fit Gamma-Pareto distribution to data using MLE

        Following the procedure in Section 5 of the paper:
        1. Estimate θ as sample minimum x_{(1)}
        2. Solve for α and c using MLE equations

        Parameters:
        -----------
        data : array-like
            Data to fit
        initial_params : tuple, optional
            Initial values for (α, c)
        method : str
            'hybrid' (default): Use equation solving for α, then compute c
            'optimize': Use numerical optimization

        Returns:
        --------
        self with fitted parameters
        """
        data = np.array(data)
        self.data = data

        # Step 1: Estimate θ as sample minimum (following Smith's method)
        self.x_min = np.min(data)
        self.theta_hat = self.x_min

        # Count frequency of minimum value
        self.n_min_freq = np.sum(data == self.x_min)

        # Filter data (exclude minimum values for MLE of α and c)
        data_filtered = data[data > self.x_min]
        n_filtered = len(data_filtered)

        if n_filtered < 2:
            raise ValueError("Insufficient data points above minimum for MLE")

        # Calculate statistics needed for estimation
        log_ratio = np.log(data_filtered / self.x_min)
        m1_star = np.mean(log_ratio)  # m1* in equation (5.5)
        log_log_ratio = np.log(log_ratio)
        m2_star = np.mean(log_log_ratio)  # m2* in equation (5.6)

        if method == 'hybrid':
            # Method 1: Solve equation (5.6) for α, then compute c from (5.5)
            # Equation (5.6): ψ(α) - log(α) + log(m1*) - m2* = 0

            def equation_for_alpha(alpha):
                return digamma(alpha) - np.log(alpha) + np.log(m1_star) - m2_star

            # Find initial guess for α
            # From the paper: α₀ = ȳ² / s_y²
            y = log_ratio
            y_mean = np.mean(y)
            y_var = np.var(y, ddof=1)
            alpha_init = y_mean ** 2 / y_var if y_var > 0 else 1.0

            # Solve for α
            try:
                alpha_solution = fsolve(equation_for_alpha, alpha_init)[0]
                if alpha_solution > 0:
                    self.alpha_hat = alpha_solution
                else:
                    # Fallback to optimization
                    method = 'optimize'
            except:
                method = 'optimize'

        if method == 'optimize':
            # Method 2: Numerical optimization of log-likelihood
            if initial_params is None:
                # Initial values from method of moments
                y = log_ratio
                y_mean = np.mean(y)
                y_var = np.var(y, ddof=1)
                c_init = y_var / y_mean if y_mean > 0 else 0.1
                alpha_init = y_mean ** 2 / y_var if y_var > 0 else 1.0
                initial_params = (alpha_init, c_init)

            # Negative log-likelihood for minimization
            def neg_log_lik(params):
                alpha, c = params
                if alpha <= 0 or c <= 0:
                    return 1e10
                return -self._log_likelihood_function(params, data, self.x_min)

            # Bounds for parameters
            bounds = [(1e-6, None), (1e-6, None)]

            # Optimize
            result = minimize(neg_log_lik, initial_params, bounds=bounds,
                              method='L-BFGS-B')

            if result.success:
                self.alpha_hat, self.c_hat = result.x
                self.log_likelihood = -result.fun
            else:
                raise RuntimeError("Optimization failed: " + result.message)

        else:  # hybrid method succeeded
            # Compute c from equation (5.5): c = m1* / α
            self.c_hat = m1_star / self.alpha_hat

            # Compute log-likelihood
            self.log_likelihood = self._log_likelihood_function(
                [self.alpha_hat, self.c_hat], data, self.x_min
            )

        # Calculate standard errors using Fisher information
        self._calculate_standard_errors(data_filtered)

        return self

    def _calculate_standard_errors(self, data_filtered):
        """
        Calculate standard errors of parameter estimates

        Using equations (5.9) and (5.10) from the paper
        """
        n = len(data_filtered)
        alpha = self.alpha_hat
        c = self.c_hat

        # Fisher information from equation (5.7)
        psi_prime = polygamma(1, alpha)  # trigamma function

        # Variance-covariance matrix from equation (5.8)
        denom = n * (alpha * psi_prime - 1)

        if denom > 0:
            var_alpha = alpha / denom
            var_c = c ** 2 * psi_prime / denom

            self.alpha_se = np.sqrt(var_alpha)
            self.c_se = np.sqrt(var_c)

            # Approximate standard errors using equation (5.10)
            self.alpha_se_approx = np.sqrt(6 * alpha ** 3 / (n * (3 * alpha + 1)))
            self.c_se_approx = np.sqrt(c ** 2 * (6 * alpha ** 2 + 3 * alpha + 1) /
                                       (n * alpha * (3 * alpha + 1)))
        else:
            self.alpha_se = np.nan
            self.c_se = np.nan
            self.alpha_se_approx = np.nan
            self.c_se_approx = np.nan

    def summary(self):
        """Print summary of fitted distribution"""
        if self.alpha_hat is None:
            print("Model not fitted yet.")
            return

        print("=" * 60)
        print("GAMMA-PARETO DISTRIBUTION - MAXIMUM LIKELIHOOD ESTIMATION")
        print("=" * 60)
        print(f"Sample size: {len(self.data)}")
        print(f"Minimum value (θ̂): {self.theta_hat:.6f}")
        print(f"Frequency of minimum: {self.n_min_freq}")
        print(f"Data points above minimum: {len(self.data[self.data > self.x_min])}")
        print()
        print("PARAMETER ESTIMATES:")
        print(f"  α (shape): {self.alpha_hat:.6f}")
        print(f"  c (scale): {self.c_hat:.6f}")
        print()
        print("STANDARD ERRORS (exact):")
        print(f"  se(α): {self.alpha_se:.6f}")
        print(f"  se(c): {self.c_se:.6f}")
        print()
        print("STANDARD ERRORS (approximate):")
        print(f"  se(α) approx: {self.alpha_se_approx:.6f}")
        print(f"  se(c) approx: {self.c_se_approx:.6f}")
        print()
        print("DISTRIBUTION PROPERTIES:")
        mode_val = self.mode(self.alpha_hat, self.c_hat, self.theta_hat)
        mean_val = self.mean(self.alpha_hat, self.c_hat, self.theta_hat)
        var_val = self.variance(self.alpha_hat, self.c_hat, self.theta_hat)

        print(f"  Mode: {mode_val:.6f}")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Variance: {var_val:.6f}")
        print(f"  Log-likelihood: {self.log_likelihood:.6f}")
        print(f"  AIC: {2 * 3 - 2 * self.log_likelihood:.6f}")
        print("=" * 60)

    def plot_fit(self, bins=50, figsize=(12, 8)):
        """Plot fitted distribution against data"""
        if self.alpha_hat is None:
            print("Model not fitted yet.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Histogram with fitted PDF
        ax1 = axes[0, 0]
        ax1.hist(self.data, bins=bins, density=True, alpha=0.6,
                 label='Data', edgecolor='black')

        x_vals = np.linspace(self.theta_hat, np.max(self.data), 1000)
        pdf_vals = self.pdf(x_vals, self.alpha_hat, self.c_hat, self.theta_hat)
        ax1.plot(x_vals, pdf_vals, 'r-', linewidth=2, label='Gamma-Pareto fit')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Density')
        ax1.set_title('PDF Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Empirical vs fitted CDF
        ax2 = axes[0, 1]
        sorted_data = np.sort(self.data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, ecdf, 'b-', alpha=0.7, label='Empirical CDF')

        cdf_vals = self.cdf(sorted_data, self.alpha_hat, self.c_hat, self.theta_hat)
        ax2.plot(sorted_data, cdf_vals, 'r-', linewidth=2, label='Fitted CDF')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('CDF Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Q-Q plot
        ax3 = axes[1, 0]
        theoretical_quantiles = self.cdf(sorted_data, self.alpha_hat,
                                         self.c_hat, self.theta_hat)
        ax3.scatter(theoretical_quantiles, ecdf, alpha=0.6)
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax3.set_xlabel('Theoretical Quantiles')
        ax3.set_ylabel('Empirical Quantiles')
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')

        # Hazard function
        ax4 = axes[1, 1]
        hazard_vals = self.hazard_function(x_vals, self.alpha_hat,
                                           self.c_hat, self.theta_hat)
        ax4.plot(x_vals, hazard_vals, 'g-', linewidth=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Hazard Rate')
        ax4.set_title('Hazard Function')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example 1: Replicate Floyd River Flood Data Analysis
def example_floyd_river():
    """Replicate analysis from Table 3 and 4 in the paper"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: FLOYD RIVER FLOOD DATA")
    print("=" * 60)

    # Floyd River flood discharge data from Table 3
    flood_data = [
        1460, 4050, 3570, 2060, 1300, 1390, 1720, 6280, 1360, 7440,
        5320, 1400, 3240, 2710, 4520, 4840, 8320, 13900, 71500, 6250,
        2260, 318, 1330, 970, 1920, 15100, 2870, 20600, 3810, 726,
        7500, 7170, 2000, 829, 17300, 4740, 13400, 2940, 5660
    ]

    print(f"Data size: {len(flood_data)}")
    print(f"Minimum value: {np.min(flood_data)}")
    print(f"Maximum value: {np.max(flood_data)}")

    # Fit Gamma-Pareto distribution
    estimator = GammaParetoMLE()
    estimator.fit(flood_data)
    estimator.summary()

    # Compare with paper results (Table 4)
    print("\nCOMPARISON WITH PAPER RESULTS (Table 4):")
    print("Parameter       Our Estimate   Paper Estimate")
    print(f"α (alpha)       {estimator.alpha_hat:.4f}        5.1454")
    print(f"c              {estimator.c_hat:.4f}        0.4712")
    print(f"θ (theta)      {estimator.theta_hat:.0f}          318")
    print(f"Log-likelihood {estimator.log_likelihood:.2f}        -365.81")
    print(f"AIC            {2 * 3 - 2 * estimator.log_likelihood:.1f}          734.9")

    # Plot the fit
    estimator.plot_fit(bins=20)

    return estimator


# Example 2: Replicate Fatigue Life Data Analysis
def example_fatigue_life():
    """Replicate analysis from Table 5 and 6 in the paper"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: FATIGUE LIFE OF 6061-T6 ALUMINUM DATA")
    print("=" * 60)

    # Fatigue life data from Table 5
    fatigue_data = [
        70, 90, 96, 97, 99, 100, 103, 104, 104, 105,
        107, 108, 108, 108, 109, 109, 112, 112, 113, 114,
        114, 114, 116, 119, 120, 120, 120, 121, 121, 123,
        124, 124, 124, 124, 124, 128, 128, 129, 129, 130,
        130, 130, 131, 131, 131, 131, 132, 132, 132, 132,
        133, 134, 134, 134, 134, 134, 136, 137, 138, 138,
        138, 139, 139, 141, 141, 142, 142, 142, 142, 142,
        142, 142, 144, 144, 145, 146, 148, 148, 149, 151,
        151, 152, 155, 156, 157, 157, 157, 158, 159, 159,
        162, 163, 163, 164, 166, 166, 168, 170, 174, 196, 212
    ]

    print(f"Data size: {len(fatigue_data)}")
    print(f"Minimum value: {np.min(fatigue_data)}")
    print(f"Maximum value: {np.max(fatigue_data)}")

    # Fit Gamma-Pareto distribution
    estimator = GammaParetoMLE()
    estimator.fit(fatigue_data)
    estimator.summary()

    # Compare with paper results (Table 6)
    print("\nCOMPARISON WITH PAPER RESULTS (Table 6):")
    print("Parameter       Our Estimate   Paper Estimate")
    print(f"α (alpha)       {estimator.alpha_hat:.4f}       15.0209")
    print(f"c              {estimator.c_hat:.5f}       0.04258")
    print(f"θ (theta)      {estimator.theta_hat:.0f}          70")
    print(f"Log-likelihood {estimator.log_likelihood:.2f}        -448.53")
    print(f"AIC            {2 * 3 - 2 * estimator.log_likelihood:.1f}          900.6")

    # Plot the fit
    estimator.plot_fit(bins=30)

    return estimator


# Example 3: Replicate Tribolium Confusum Data Analysis
def example_tribolium():
    """Replicate analysis from Table 7 and 8 in the paper"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: TRIBOLIUM CONFUSUM STRAIN #3 DATA")
    print("=" * 60)

    # Create data from frequency table (Table 7)
    # x-values with frequencies
    x_values = [55, 65, 75, 85, 95, 105, 115, 125, 135, 145,
                155, 165, 175, 185, 195, 205, 215, 225, 235, 245]
    frequencies = [3, 20, 53, 78, 86, 86, 68, 51, 20, 11,
                   6, 4, 7, 5, 1, 2, 0, 1, 1, 1]

    # Expand to individual data points
    tribolium_data = []
    for x, freq in zip(x_values, frequencies):
        tribolium_data.extend([x] * freq)

    tribolium_data = np.array(tribolium_data)

    print(f"Data size: {len(tribolium_data)}")
    print(f"Minimum value: {np.min(tribolium_data)}")
    print(f"Maximum value: {np.max(tribolium_data)}")

    # Fit Gamma-Pareto distribution
    estimator = GammaParetoMLE()
    estimator.fit(tribolium_data)
    estimator.summary()

    # Compare with paper results (Table 8)
    print("\nCOMPARISON WITH PAPER RESULTS (Table 8):")
    print("Parameter       Our Estimate   Paper Estimate")
    print(f"α (alpha)       {estimator.alpha_hat:.4f}        6.3513")
    print(f"c              {estimator.c_hat:.5f}       0.09743")
    print(f"θ (theta)      {estimator.theta_hat:.0f}          55")
    print(f"Log-likelihood {estimator.log_likelihood:.2f}        -2297.70")
    print(f"AIC            {2 * 3 - 2 * estimator.log_likelihood:.1f}          4599.4")

    # Plot the fit
    estimator.plot_fit(bins=20)

    return estimator


# Additional utility functions
def generate_gamma_pareto_data(n, alpha, c, theta, random_state=None):
    """
    Generate random samples from Gamma-Pareto distribution

    Using Lemma 1: If Y ~ Gamma(α, c), then X = θ * exp(Y) ~ Gamma-Pareto(α, c, θ)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate from Gamma distribution
    y = np.random.gamma(shape=alpha, scale=c, size=n)

    # Transform to Gamma-Pareto
    x = theta * np.exp(y)

    return x


def goodness_of_fit_test(estimator, n_bins=None):
    """
    Perform chi-squared goodness-of-fit test
    """
    from scipy.stats import chi2

    data = estimator.data
    alpha = estimator.alpha_hat
    c = estimator.c_hat
    theta = estimator.theta_hat

    if n_bins is None:
        n_bins = int(np.sqrt(len(data)))

    # Create bins
    bins = np.linspace(theta, np.max(data), n_bins + 1)

    # Observed frequencies
    observed, _ = np.histogram(data, bins=bins)

    # Expected frequencies
    cdf_vals = estimator.cdf(bins, alpha, c, theta)
    expected = len(data) * np.diff(cdf_vals)

    # Chi-squared test (combine bins with small expected counts)
    valid_mask = expected > 5
    if np.sum(valid_mask) < 3:
        # Combine bins
        min_expected = 5
        combined_obs = []
        combined_exp = []
        cum_obs = 0
        cum_exp = 0

        for i in range(len(expected)):
            cum_obs += observed[i]
            cum_exp += expected[i]

            if cum_exp >= min_expected or i == len(expected) - 1:
                combined_obs.append(cum_obs)
                combined_exp.append(cum_exp)
                cum_obs = 0
                cum_exp = 0

        observed = np.array(combined_obs)
        expected = np.array(combined_exp)
        valid_mask = expected > 0
    else:
        observed = observed[valid_mask]
        expected = expected[valid_mask]

    # Chi-squared statistic
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 3  # 3 parameters estimated
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, df, p_value


# Run all examples
if __name__ == "__main__":
    print("GAMMA-PARETO DISTRIBUTION MLE IMPLEMENTATION")
    print("Based on: Alzaatreh, Famoye & Lee (2012)")
    print("Gamma-Pareto Distribution and Its Applications")
    print()

    # Run the three examples from the paper
    estimator1 = example_floyd_river()
    estimator2 = example_fatigue_life()
    estimator3 = example_tribolium()

    print("\n" + "=" * 60)
    print("SIMULATION EXAMPLE")
    print("=" * 60)

    # Generate and fit simulated data
    np.random.seed(42)
    true_alpha, true_c, true_theta = 3.0, 0.2, 10.0
    sim_data = generate_gamma_pareto_data(1000, true_alpha, true_c, true_theta)

    print(f"\nTrue parameters: α={true_alpha}, c={true_c}, θ={true_theta}")

    estimator_sim = GammaParetoMLE()
    estimator_sim.fit(sim_data)
    estimator_sim.summary()

    # Goodness-of-fit test
    chi2_stat, df, p_value = goodness_of_fit_test(estimator_sim)
    print(f"\nGoodness-of-fit test:")
    print(f"  Chi-squared: {chi2_stat:.4f}")
    print(f"  Degrees of freedom: {df}")
    print(f"  p-value: {p_value:.4f}")

    # Confidence intervals
    alpha_ci = (
        estimator_sim.alpha_hat - 1.96 * estimator_sim.alpha_se,
        estimator_sim.alpha_hat + 1.96 * estimator_sim.alpha_se
    )
    c_ci = (
        estimator_sim.c_hat - 1.96 * estimator_sim.c_se,
        estimator_sim.c_hat + 1.96 * estimator_sim.c_se
    )

    print(f"\n95% Confidence Intervals:")
    print(f"  α: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}]")
    print(f"  c: [{c_ci[0]:.4f}, {c_ci[1]:.4f}]")