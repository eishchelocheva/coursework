import numpy as np
import matplotlib.pyplot as plt

from sir_abc_utils import (
    simulate_sir,
    generate_observations,
    run_abc_simulations,
    analyze_thresholds,
)


np.random.seed(42)


def print_results(results_table):
    print("\nInfluence of n_simulations")
    print("-" * 100)
    print(
        "n_sim | epsilon | accepted | acc_rate | "
        "beta1_est | beta2_est | beta3_est | "
        "beta1_err | beta2_err | beta3_err"
    )
    print("-" * 100)

    for r in results_table:
        print(
            f"{r['n_simulations']:<5d} | "
            f"{r['epsilon']:<7.4f} | "
            f"{r['accepted_count']:<8d} | "
            f"{r['acceptance_rate']:<8.3f} | "
            f"{r['beta1_est']:<9.4f} | "
            f"{r['beta2_est']:<9.4f} | "
            f"{r['beta3_est']:<9.4f} | "
            f"{r['beta1_error']:<9.4f} | "
            f"{r['beta2_error']:<9.4f} | "
            f"{r['beta3_error']:<9.4f}"
        )


def plot_errors_vs_n(results_table):
    n_values = [r["n_simulations"] for r in results_table]

    plt.figure(figsize=(10, 6))

    plt.plot(
        n_values,
        [r["beta1_error"] for r in results_table],
        marker="o",
        label="Ошибка beta1",
    )
    plt.plot(
        n_values,
        [r["beta2_error"] for r in results_table],
        marker="o",
        label="Ошибка beta2",
    )
    plt.plot(
        n_values,
        [r["beta3_error"] for r in results_table],
        marker="o",
        label="Ошибка beta3",
    )

    plt.xlabel("Число симуляций")
    plt.ylabel("Абсолютная ошибка")
    plt.title("Влияние числа симуляций на ошибку восстановления параметров")
    plt.legend()
    plt.grid(True)

    plt.savefig("n_simulations_errors.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_estimates_vs_n(results_table, beta1_true, beta2_true, beta3_true):
    n_values = [r["n_simulations"] for r in results_table]

    plt.figure(figsize=(10, 6))

    plt.plot(
        n_values,
        [r["beta1_est"] for r in results_table],
        marker="o",
        label="Оценка beta1",
    )
    plt.plot(
        n_values,
        [r["beta2_est"] for r in results_table],
        marker="o",
        label="Оценка beta2",
    )
    plt.plot(
        n_values,
        [r["beta3_est"] for r in results_table],
        marker="o",
        label="Оценка beta3",
    )

    plt.axhline(beta1_true, linestyle="--", label="Истинное beta1")
    plt.axhline(beta2_true, linestyle="--", color = 'orange', label="Истинное beta2")
    plt.axhline(beta3_true, linestyle="--", color = 'green', label="Истинное beta3")

    plt.xlabel("Число симуляций")
    plt.ylabel("Значение параметра")
    plt.title("Оценки параметров при разном числе симуляций")
    plt.legend()
    plt.grid(True)

    plt.savefig("n_simulations_estimates.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_histograms_for_n(result, beta1_true, beta2_true, beta3_true):
    accepted_params = result["accepted_params"]
    n_simulations = result["n_simulations"]

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

    fig.suptitle(
        f"Апостериорные распределения при n_simulations = {n_simulations}",
        fontsize=14,
    )

    plt.tight_layout()

    filename = f"n_simulations_posteriors_{n_simulations}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


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
acceptance_quantile = 0.02

n_simulations_values = [1000, 3000, 5000, 10000]


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


results_table = []


for n_simulations in n_simulations_values:
    print(f"\nRunning ABC with n_simulations = {n_simulations}")

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

    results = analyze_thresholds(
        all_params=all_params,
        all_distances=all_distances,
        beta_true=beta_true,
        quantiles=[acceptance_quantile],
        n_simulations=n_simulations,
    )

    result = results[0]
    result["n_simulations"] = n_simulations

    results_table.append(result)


print_results(results_table)

plot_errors_vs_n(results_table)
plot_estimates_vs_n(results_table, beta1_true, beta2_true, beta3_true)

for result in results_table:
    plot_histograms_for_n(result, beta1_true, beta2_true, beta3_true)