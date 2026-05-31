import numpy as np
import matplotlib.pyplot as plt

from sir_abc_utils import (
    simulate_sir,
    generate_observations,
)


np.random.seed(42)


def summary_peak_time_load(t, I):
    """
    Базовый набор сводных статистик:
    1) максимум I(t),
    2) время достижения пика,
    3) сумма значений I(t) за весь период.
    """
    return np.array([
        np.max(I),
        t[np.argmax(I)],
        np.sum(I),
    ])


def summary_peak_time_final(t, I):
    """
    Набор статистик:
    1) максимум I(t),
    2) время достижения пика,
    3) значение I(t) в конце периода.
    """
    return np.array([
        np.max(I),
        t[np.argmax(I)],
        I[-1],
    ])


def summary_peak_load_final(t, I):
    """
    Набор статистик:
    1) максимум I(t),
    2) сумма значений I(t),
    3) значение I(t) в конце периода.
    """
    return np.array([
        np.max(I),
        np.sum(I),
        I[-1],
    ])


def summary_quantiles(t, I):
    """
    Набор статистик на основе квантилей распределения значений I(t).
    """
    return np.array([
        np.quantile(I, 0.25),
        np.quantile(I, 0.50),
        np.quantile(I, 0.75),
    ])


def summary_mean_std_peak(t, I):
    """
    Набор статистик:
    1) среднее значение I(t),
    2) стандартное отклонение I(t),
    3) максимум I(t).
    """
    return np.array([
        np.mean(I),
        np.std(I),
        np.max(I),
    ])


SUMMARY_FUNCTIONS = {
    "peak_time_load": summary_peak_time_load,
    "peak_time_final": summary_peak_time_final,
    "peak_load_final": summary_peak_load_final,
    "quantiles": summary_quantiles,
    "mean_std_peak": summary_mean_std_peak,
}

def distance_summary_stats(stats_sim, stats_obs):
    """
    Нормированное евклидово расстояние между сводными статистиками.
    Нормировка нужна, потому что разные статистики имеют разные масштабы.
    """
    scale = np.maximum(np.abs(stats_obs), 1.0)
    normalized_diff = (stats_sim - stats_obs) / scale

    return np.sqrt(np.sum(normalized_diff ** 2))


def run_abc_with_summary_statistics(
    I_obs,
    sampling_rate,
    n_simulations,
    summary_function,
    gamma,
    N,
    I0,
    R_initial,
    T,
    t1,
    t2,
    num_points,
    beta1_prior=(0.05, 0.5),
    beta2_prior=(0.01, 0.3),
    beta3_prior=(0.05, 0.5),
):
    """
    Запускает ABC-симуляции для заданного набора сводных статистик.
    """
    all_params = []
    all_distances = []

    t_obs = np.linspace(0, T, num_points)
    stats_obs = summary_function(t_obs, I_obs)

    for i in range(n_simulations):
        if i % 100 == 0:
            print(f"Simulation {i}/{n_simulations}")

        beta1_candidate = np.random.uniform(*beta1_prior)
        beta2_candidate = np.random.uniform(*beta2_prior)
        beta3_candidate = np.random.uniform(*beta3_prior)

        t_sim, _, I_sim, _ = simulate_sir(
            beta1=beta1_candidate,
            beta2=beta2_candidate,
            beta3=beta3_candidate,
            gamma=gamma,
            N=N,
            I0=I0,
            R_initial=R_initial,
            T=T,
            t1=t1,
            t2=t2,
            num_points=num_points,
        )

        I_sim_obs = generate_observations(
            I_sim,
            sampling_rate=sampling_rate,
            random_seed=None,
        )

        stats_sim = summary_function(t_sim, I_sim_obs)
        distance = distance_summary_stats(stats_sim, stats_obs)

        all_params.append([
            beta1_candidate,
            beta2_candidate,
            beta3_candidate,
        ])
        all_distances.append(distance)

    return np.array(all_params), np.array(all_distances)


def analyze_one_summary_set(
    summary_name,
    summary_function,
    I_obs,
    sampling_rate,
    n_simulations,
    acceptance_quantile,
    beta_true,
    gamma_true,
    N,
    I0,
    R_initial,
    T,
    t1,
    t2,
    num_points,
):
    """
    Запускает ABC для одного набора сводных статистик
    и возвращает результаты.
    """
    print(f"\nRunning ABC for summary statistics: {summary_name}")

    all_params, all_distances = run_abc_with_summary_statistics(
        I_obs=I_obs,
        sampling_rate=sampling_rate,
        n_simulations=n_simulations,
        summary_function=summary_function,
        gamma=gamma_true,
        N=N,
        I0=I0,
        R_initial=R_initial,
        T=T,
        t1=t1,
        t2=t2,
        num_points=num_points,
    )

    epsilon = np.quantile(all_distances, acceptance_quantile)
    accepted_mask = all_distances <= epsilon
    accepted_params = all_params[accepted_mask]

    beta_estimates = accepted_params.mean(axis=0)
    errors = np.abs(beta_estimates - beta_true)

    return {
        "summary_name": summary_name,
        "epsilon": epsilon,
        "accepted_count": len(accepted_params),
        "acceptance_rate": len(accepted_params) / n_simulations,
        "beta1_est": beta_estimates[0],
        "beta2_est": beta_estimates[1],
        "beta3_est": beta_estimates[2],
        "beta1_error": errors[0],
        "beta2_error": errors[1],
        "beta3_error": errors[2],
        "accepted_params": accepted_params,
    }


def print_results(results_table):
    print("\nInfluence of summary statistics")
    print("-" * 120)
    print(
        "summary | epsilon | accepted | acc_rate | "
        "beta1_est | beta2_est | beta3_est | "
        "beta1_err | beta2_err | beta3_err"
    )
    print("-" * 120)

    for r in results_table:
        print(
            f"{r['summary_name']:<16s} | "
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


def plot_errors_vs_summary(results_table):
    summary_names = [r["summary_name"] for r in results_table]

    plt.figure(figsize=(12, 6))

    plt.plot(
        summary_names,
        [r["beta1_error"] for r in results_table],
        marker="o",
        label="Ошибка beta1",
    )
    plt.plot(
        summary_names,
        [r["beta2_error"] for r in results_table],
        marker="o",
        label="Ошибка beta2",
    )
    plt.plot(
        summary_names,
        [r["beta3_error"] for r in results_table],
        marker="o",
        label="Ошибка beta3",
    )

    plt.xlabel("Набор сводных статистик")
    plt.ylabel("Абсолютная ошибка")
    plt.title("Влияние выбора сводных статистик на ошибку восстановления параметров")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=20)

    plt.savefig("summary_statistics_errors.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_estimates_vs_summary(results_table, beta1_true, beta2_true, beta3_true):
    summary_names = [r["summary_name"] for r in results_table]

    plt.figure(figsize=(12, 6))

    plt.plot(
        summary_names,
        [r["beta1_est"] for r in results_table],
        marker="o",
        label="Оценка beta1",
    )
    plt.plot(
        summary_names,
        [r["beta2_est"] for r in results_table],
        marker="o",
        label="Оценка beta2",
    )
    plt.plot(
        summary_names,
        [r["beta3_est"] for r in results_table],
        marker="o",
        label="Оценка beta3",
    )

    plt.axhline(beta1_true, linestyle="--", label="Истинное beta1")
    plt.axhline(beta2_true, linestyle="--", color="orange", label="Истинное beta2")
    plt.axhline(beta3_true, linestyle="--", color="green", label="Истинное beta3")

    plt.xlabel("Набор сводных статистик")
    plt.ylabel("Значение параметра")
    plt.title("Оценки параметров при разных наборах сводных статистик")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=20)

    plt.savefig("summary_statistics_estimates.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_histograms_for_summary(result, beta1_true, beta2_true, beta3_true):
    accepted_params = result["accepted_params"]
    summary_name = result["summary_name"]

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
        f"Апостериорные распределения для статистик: {summary_name}",
        fontsize=14,
    )

    plt.tight_layout()

    filename = f"summary_statistics_posteriors_{summary_name}.png"
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

# Для исследования summary statistics фиксируем полную наблюдаемость, чтобы меньше смешивать эффект выбора статистик со стохастическим шумом наблюдения.
sampling_rate = 1.0
n_simulations = 3000
acceptance_quantile = 0.02


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

for summary_name, summary_function in SUMMARY_FUNCTIONS.items():
    result = analyze_one_summary_set(
        summary_name=summary_name,
        summary_function=summary_function,
        I_obs=I_obs,
        sampling_rate=sampling_rate,
        n_simulations=n_simulations,
        acceptance_quantile=acceptance_quantile,
        beta_true=beta_true,
        gamma_true=gamma_true,
        N=N,
        I0=I0,
        R_initial=R_initial,
        T=T,
        t1=t1,
        t2=t2,
        num_points=num_points,
    )

    results_table.append(result)

print_results(results_table)

plot_errors_vs_summary(results_table)
plot_estimates_vs_summary(results_table, beta1_true, beta2_true, beta3_true)

for result in results_table:
    plot_histograms_for_summary(result, beta1_true, beta2_true, beta3_true)
