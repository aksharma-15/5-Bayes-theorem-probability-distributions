# 5-Bayes-theorem-probability-distributions
Application of bayes' theorem and distributions for data science

---

# Stochastic Ride-Hailing Analytics: Probability, Bayesian Inference & Statistical Distributions

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

## Overview

This repository presents an applied statistical analysis of a synthetic one-month ride-hailing dataset (`N = 4,328` simulated rides across `720` hourly slots). Written from an applied machine learning engineering perspective, the notebook connects foundational probability theory to production operational modeling—covering conditional cancellation risk, Bayesian fraud posterior updates, Gaussian SLA capacity planning, and Binomial shift cancellation forecasting.

---

## Synthetic Dataset Schema (`load_uber`)

The dataset is generated via a deterministic pseudo-random number generator (`np.random.default_rng(seed=11)`) modeling a 30-day (`720` hourly intervals) operational window:

| Column | Data Type | Generative Process & Domain Logic |
| :--- | :--- | :--- |
| `hour_slot` | `int64` | Sequential hourly index (`0` to `719`), populated via a Poisson arrival process (`lambda = 6` rides/hour). |
| `hour` | `int64` | Diurnal hour of the day (`hour_slot % 24`, ranging `0–23`). |
| `is_peak` | `int64` | Binary indicator (`1` if `hour` in `[8, 9, 10, 17, 18, 19]`, else `0`) capturing morning and evening rush hours. |
| `trip_minutes` | `float64` | Continuous trip duration drawn from a truncated Normal distribution (`mu = 24`, `sigma = 6`, lower-bounded at `4.0` mins). |
| `fare` | `float64` | Base linear pricing (`45 + 11 * trip_minutes + Normal(0, 25)`), scaled by a `Uniform(2.0, 2.8)` multiplier during surge events. |
| `surge` | `int64` | Bernoulli indicator (`p = 0.02`) representing dynamic surge pricing activation. |
| `cancelled` | `int64` | Conditional Bernoulli outcome: `P(cancelled \| is_peak=1) = 0.20` vs. `P(cancelled \| is_peak=0) = 0.08`. |
| `is_fraud` | `int64` | Ground-truth Bernoulli class label (`p = 0.01`) modeling severe class imbalance. |
| `flagged` | `int64` | Heuristic/model alert trigger: `90%` True Positive Rate (Recall) on fraudulent rides and `5%` False Positive Rate on legitimate rides. |

---

## Core Mathematical & Statistical Concepts

### 1. Empirical Probability & Conditional Asymmetry

We compute marginal and conditional probabilities directly from empirical sample frequencies:

* **Marginal (Unconditional) Probability:** The baseline probability of an event across the entire sample space (`469` cancellations out of `4,328` rides):
  ```text
  P(Cancelled) = 469 / 4328 = 0.108 (10.8%)
  ```
* **Conditional Probability & Conditioning Direction:** Conditioning restricts the sample space to a specific event subset. While **joint probability** is commutative (`P(A and B) = P(B and A)`), **conditional probability is strictly asymmetric** (`P(A | B) != P(B | A)`) whenever `P(A) != P(B)`:
  * **Forward Risk (`P(Cancelled | Peak)`):** Conditioning on the `1,064` rush-hour rides, the cancellation probability rises from `0.108` to **`0.178`** (`17.8%`).
  * **Inverse Attribution (`P(Peak | Cancelled)`):** Conditioning on the `469` cancelled rides, the probability that a cancellation occurred during rush hour is **`0.403`** (`40.3%`).

---

### 2. Bayesian Inference & Base-Rate Fallacy in Fraud Detection

In heavily imbalanced classification problems (such as fraud detection, spam filtering, or medical screening), high model sensitivity does not guarantee high precision. Out of `4,328` total rides, only `44` are genuinely fraudulent (`1.02%` prevalence), yet the alerting system flags `270` rides.

Applying **Bayes' Theorem**, we update our prior belief to compute the posterior probability that a flagged ride is actually fraudulent:

```text
P(Fraud | Flagged) = [ P(Flagged | Fraud) * P(Fraud) ] / P(Flagged)
```

Where the evidence (marginal likelihood) `P(Flagged)` is expanded via the **Law of Total Probability**:

```text
P(Flagged) = P(Flagged | Fraud) * P(Fraud) + P(Flagged | No Fraud) * P(No Fraud)
```

| Bayesian Term | Mathematical Notation | Empirical Value | ML / Statistical Interpretation |
| :--- | :--- | :--- | :--- |
| **Prior** | `P(Fraud)` | `0.0102` | Baseline class prevalence before observing any system alert. |
| **Likelihood (Sensitivity / TPR)** | `P(Flagged \| Fraud)` | `0.909` | Recall of the detector on true fraud instances. |
| **False Alarm Rate (FPR)** | `P(Flagged \| No Fraud)` | `0.054` | Type I error rate on legitimate rides (`1 - Specificity`). |
| **Evidence (Marginal Likelihood)** | `P(Flagged)` | `0.0624` | Overall alert rate (`0.909 * 0.0102 + 0.054 * 0.9898`). |
| **Posterior (Precision / PPV)** | `P(Fraud \| Flagged)` | **`0.148`** | Only **`14.8%`** of flagged rides are truly fraudulent due to the low base rate. |

> **Production Takeaway:** Even with `~90.9%` recall and `~94.6%` specificity, **`85.2%`** (`1 - 0.148`) of system alerts are false positives. Production fraud pipelines must calibrate decision thresholds against base-rate priors and downstream manual review costs.

---

### 3. Continuous Distributions: Gaussian Symmetry vs. Right-Skewed Heavy Tails

#### A. Normal (Gaussian) Distribution (`trip_minutes`)
Trip duration follows a symmetric, unimodal bell curve parameterized by empirical mean `mu = 24.12` minutes and sample standard deviation `sigma = 6.08` minutes:

* **Empirical Rule (`68–95–99.7` / Three-Sigma Rule):** Sample quantiles closely match theoretical Gaussian bounds:
  * `mu +/- 1 * sigma`: **`68.3%`** of trips (Theoretical: `68.27%`)
  * `mu +/- 2 * sigma`: **`95.3%`** of trips (Theoretical: `95.45%`)
  * `mu +/- 3 * sigma`: **`99.8%`** of trips (Theoretical: `99.73%`)
* **Cumulative Distribution Function (`CDF`):** Evaluates `P(X <= x)`—the probability that a ride finishes within a target SLA threshold:
  * `P(Trip <= 30 mins)` = `stats.norm.cdf(30, loc=mu, scale=sig)` = **`0.8330`** (`83.3%`)
  * `P(Trip <= 32 mins)` = `stats.norm.cdf(32, loc=mu, scale=sig)` = **`0.9023`** (`90.2%`)
  * **Survival Function (`1 - CDF`):** `P(Trip > 38 mins)` = **`0.011`** (`1.1%` tail risk)
* **Percent Point Function (`PPF` / Quantile Function):** The mathematical inverse of the `CDF`, solving for duration `x` given a target probability `q`:
  * **95th Percentile SLA (`q = 0.95`):** `stats.norm.ppf(0.95, loc=mu, scale=sig)` = **`34.13` minutes**. Using this conservative 95th-percentile trip time (`60 / 34.13`), a single driver is guaranteed to clear at least **`1.76` rides/hour** at the 95% confidence level.
  * **99.999th Percentile Tail (`q = 0.99999`):** **`50.07` minutes** (capturing extreme multi-sigma tail events).

#### B. Skewness & Parametric Mismatch (`fare`)
Unlike `trip_minutes`, `fare` exhibits strong **positive (right) skewness (`skewness = 2.639`)** driven by multiplicative surge pricing (`2.0x–2.8x`). Because the arithmetic mean (`318.45`) is pulled toward the heavy right tail, Gaussian assumptions (`mean +/- std`) fail to represent typical fares, necessitating non-parametric quantiles (median/IQR) or log-normal transformations.

---

### 4. Discrete Distributions: Binomial Shift Modeling

To model the discrete count of cancelled rides `k` over a fixed driver shift of `n = 20` independent ride requests with constant cancellation probability `p = 0.108`, we apply the **Binomial Distribution** `X ~ Binomial(n=20, p=0.108)`:

```text
P(X = k) = C(n, k) * (p^k) * ((1 - p)^(n - k))
```

* **Probability Mass Function (`PMF` — Exact Outcomes):**
  * `P(X = 1)` = `stats.binom.pmf(1, 20, 0.108)` = **`0.245`** (`24.5%`)
  * `P(X = 2)` = `stats.binom.pmf(2, 20, 0.108)` = **`0.283`** (`28.3%` — modal outcome)
  * `P(X = 3)` = `stats.binom.pmf(3, 20, 0.108)` = **`0.206`** (`20.6%`)
* **Cumulative Distribution Function (`CDF` — Threshold Risk):**
  * `P(X <= 2)` = `stats.binom.cdf(2, 20, 0.108)` = **`0.629`** (`62.9%` probability that a driver experiences 2 or fewer cancellations in a 20-ride shift).

---

## Visualization Suite

The notebook implements reusable Matplotlib visualization wrappers with clean styling (despined top/right borders, subtle horizontal gridlines, and direct value annotations):

* `show_hist()`: Renders frequency histograms overlaid with vertical reference markers (e.g., sample mean) to visually contrast symmetric bell curves against right-skewed distributions.
* `show_bars()`: Plots conditional probabilities across categorical segments (`off-peak` vs. `rush hour`) against a dashed marginal baseline (`0.108`).
* `show_fit()`: Overlays empirical normalized value counts (bars) with theoretical probability predictions (point markers) for distribution goodness-of-fit checks.

---

## Code Annotations & Explanatory Comments

Several inline comments in the notebook highlight critical statistical operations and can be referenced directly when navigating the code:

* `per_slot = rng.poisson(6, slots)` — Models discrete event arrival rates per unit time using a Poisson process.
* `evidence = hit_rate * prior + false_alarm * (1-prior)` — Implements the Law of Total Probability to compute the normalizing denominator `P(Flagged)` in Bayes' theorem.
* `stats.norm.cdf(32, loc=mu, scale=sig)` — Evaluates the area under the Gaussian probability density curve up to `x = 32` (left-tail cumulative probability).
* `stats.norm.ppf(0.95, loc=mu, scale=sig)` — Computes the Percent Point Function (quantile function), inverting the CDF to find the SLA cutoff for `95%` of trips.
* `stats.binom.cdf(2, n_rides, p_cancel)` — Sums discrete Binomial PMF probabilities `P(X = 0) + P(X = 1) + P(X = 2)` to evaluate cumulative shift cancellation risk.

---

## Quickstart & Reproducibility

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy jupyter
   ```
2. **Execute Notebook:**
   Open `5_bayes'_theorem_and_distributions.ipynb` in JupyterLab, VS Code, or Google Colab and run all cells sequentially. All stochastic simulations are seeded via `np.random.default_rng(11)` for deterministic reproducibility.
