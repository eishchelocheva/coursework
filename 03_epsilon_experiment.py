import numpy as np
import matplotlib.pyplot as plt

from sir_abc_utils import (
    simulate_sir,
    generate_observations,
    run_abc_simulations,
    analyze_thresholds,
)


np.random.seed(42)


def print_results(results):
    print("\nSummary of threshold experiments")
    print("-" * 100)
    print(
        "quantile | epsilon | accepted | acc_rate | "
        "beta1_est | beta2_est | beta3_est | "
        "beta1_err | beta2_err | beta3_err"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['quantile']:<8.2f} | "
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


def plot_parameter_errors(results):
    quantile_labels = [str(r["quantile"]) for r in results]

    plt.figure(figsize=(10, 6))

    plt.plot(
        quantile_labels,
        [r["beta1_error"] for r in results],
        marker="o",
        label="Ошибка beta1",
    )
    plt.plot(
        quantile_labels,
        [r["beta2_error"] for r in results],
        marker="o",
        label="Ошибка beta2",
    )
    plt.plot(
        quantile_labels,
        [r["beta3_error"] for r in results],
        marker="o",
        label="Ошибка beta3",
    )

    plt.xlabel("Доля принятых симуляций")
    plt.ylabel("Абсолютная ошибка")
    plt.title("Зависимость ошибки восстановления параметров от порога принятия")
    plt.legend()
    plt.grid(True)

    plt.savefig("epsilon_parameter_errors.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_parameter_estimates(results, beta1_true, beta2_true, beta3_true):
    quantile_labels = [str(r["quantile"]) for r in results]

    plt.figure(figsize=(10, 6))

    plt.plot(
        quantile_labels,
        [r["beta1_est"] for r in results],
        marker="o",
        label="Оценка beta1",
    )
    plt.plot(
        quantile_labels,
        [r["beta2_est"] for r in results],
        marker="o",
        label="Оценка beta2",
    )
    plt.plot(
        quantile_labels,
        [r["beta3_est"] for r in results],
        marker="o",
        label="Оценка beta3",
    )

    plt.axhline(beta1_true, linestyle="--", label="Истинное beta1")
    plt.axhline(beta2_true, linestyle="--", color = "orange", label="Истинное beta2")
    plt.axhline(beta3_true, linestyle="--", color = "green", label="Истинное beta3")

    plt.xlabel("Доля принятых симуляций")
    plt.ylabel("Значение параметра")
    plt.title("Оценки параметров при разных порогах принятия")
    plt.legend()
    plt.grid(True)

    plt.savefig("epsilon_parameter_estimates.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_histograms_for_result(result, beta1_true, beta2_true, beta3_true):
    accepted_params = result["accepted_params"]
    q = result["quantile"]

    if len(accepted_params) == 0:
        print(f"No accepted samples for quantile {q}")
        return

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
        f"Апостериорные распределения при доле принятия {q}",
        fontsize=14,
    )

    plt.tight_layout()

    filename = f"epsilon_posteriors_q_{str(q).replace('.', '_')}.png"
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

quantiles = [0.20, 0.10, 0.05, 0.02, 0.01]

results = analyze_thresholds(
    all_params=all_params,
    all_distances=all_distances,
    beta_true=beta_true,
    quantiles=quantiles,
    n_simulations=n_simulations,
)

print_results(results)

plot_parameter_errors(results)
plot_parameter_estimates(results, beta1_true, beta2_true, beta3_true)

for result in results:
    plot_histograms_for_result(result, beta1_true, beta2_true, beta3_true)