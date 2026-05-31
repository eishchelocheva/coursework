import numpy as np
import matplotlib.pyplot as plt

from sir_abc_utils import (
    simulate_sir,
    generate_observations,
    run_abc_simulations,
    analyze_thresholds,
)


np.random.seed(42)


beta1_true = 0.25
beta2_true = 0.08
beta3_true = 0.22
gamma_true = 0.1

beta_true = np.array([
    beta1_true,
    beta2_true,
    beta3_true,
])

N = 10_000
I0 = 10
R_initial = 0

T = 140
t1 = 20
t2 = 60
num_points = 141

sampling_rate = 0.5
n_simulations = 1000


t, S, I, R = simulate_sir(
    beta1=beta1_true,
    beta2=beta2_true,
    beta3=beta3_true,
    gamma=gamma_true,
    N=N,
    I0=I0,
    R_initial=R_initial,
    T=T,
    t1=t1,
    t2=t2,
    num_points=num_points,
)

I_obs = generate_observations(
    I,
    sampling_rate=sampling_rate,
    random_seed=42,
)

all_params, all_distances = run_abc_simulations(
    I_obs=I_obs,
    sampling_rate=sampling_rate,
    n_simulations=n_simulations,
    gamma=gamma_true,
    N=N,
    I0=I0,
    R_initial=R_initial,
    T=T,
    t1=t1,
    t2=t2,
    num_points=num_points,
)

# Один выбранный порог: лучшие 10% симуляций
quantiles = [0.10]

results = analyze_thresholds(
    all_params=all_params,
    all_distances=all_distances,
    beta_true=beta_true,
    quantiles=quantiles,
    n_simulations=n_simulations,
)

result = results[0]
accepted_params = result["accepted_params"]

print("Quantile:", result["quantile"])
print("Epsilon:", result["epsilon"])
print("Number of accepted samples:", result["accepted_count"])
print("Acceptance rate:", result["acceptance_rate"])

print("True beta1:", beta1_true)
print("Estimated beta1:", result["beta1_est"])

print("True beta2:", beta2_true)
print("Estimated beta2:", result["beta2_est"])

print("True beta3:", beta3_true)
print("Estimated beta3:", result["beta3_est"])


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(accepted_params[:, 0], bins=30)
axes[0].axvline(beta1_true, linestyle="--", label="Истинное значение")
axes[0].set_title(r"$\beta_1$")
axes[0].set_xlabel(r"$\beta_1$")
axes[0].set_ylabel("Частота")
axes[0].legend()
axes[0].grid(True)

axes[1].hist(accepted_params[:, 1], bins=30)
axes[1].axvline(beta2_true, linestyle="--", label="Истинное значение")
axes[1].set_title(r"$\beta_2$")
axes[1].set_xlabel(r"$\beta_2$")
axes[1].set_ylabel("Частота")
axes[1].legend()
axes[1].grid(True)

axes[2].hist(accepted_params[:, 2], bins=30)
axes[2].axvline(beta3_true, linestyle="--", label="Истинное значение")
axes[2].set_title(r"$\beta_3$")
axes[2].set_xlabel(r"$\beta_3$")
axes[2].set_ylabel("Частота")
axes[2].legend()
axes[2].grid(True)

fig.suptitle("Апостериорные распределения параметров для базового запуска")
plt.tight_layout()

plt.savefig("abc_basic_posterior.png", dpi=300, bbox_inches="tight")
plt.show()