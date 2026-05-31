import numpy as np
import matplotlib.pyplot as plt

from sir_abc_utils import (
    simulate_sir,
    generate_observations,
)


np.random.seed(42)


def summary_peak_load_final(t, I):
    """
    Рабочий набор сводных статистик, выбранный на предыдущем этапе:
    1) максимум I(t),
    2) сумма значений I(t) за весь период,
    3) значение I(t) в конце периода.
    """
    return np.array([
        np.max(I),
        np.sum(I),
        I[-1],
    ])


def distance_summary_stats(stats_sim, stats_obs):
    """
    Нормированное евклидово расстояние между сводными статистиками.
    Нормировка нужна, потому что разные статистики имеют разные масштабы.
    """
    scale = np.maximum(np.abs(stats_obs), 1.0)
    normalized_diff = (stats_sim - stats_obs) / scale

    return np.sqrt(np.sum(normalized_diff ** 2))


def run_abc(
    I_obs,
    sampling_rate,
    n_simulations,
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
    Запускает ABC-симуляции при фиксированном наборе сводных статистик
    peak_load_final.
    """
    all_params = []
    all_distances = []

    t_obs = np.linspace(0, T, num_points)
    stats_obs = summary_peak_load_final(t_obs, I_obs)

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

        stats_sim = summary_peak_load_final(t_sim, I_sim_obs)
        distance = distance_summary_stats(stats_sim, stats_obs)

        all_params.append([
            beta1_candidate,
            beta2_candidate,
            beta3_candidate,
        ])
        all_distances.append(distance)

    return np.array(all_params), np.array(all_distances)


def analyze_ratios(
    all_params,
    all_distances,
    acceptance_quantile,
    beta1_true,
    beta2_true,
    beta3_true,
):
    """
    Отбирает принятые параметры и анализирует отношения beta1/beta2 и beta3/beta2.
    """
    epsilon = np.quantile(all_distances, acceptance_quantile)
    accepted_mask = all_distances <= epsilon
    accepted_params = all_params[accepted_mask]

    beta1_samples = accepted_params[:, 0]
    beta2_samples = accepted_params[:, 1]
    beta3_samples = accepted_params[:, 2]

    ratio_12_samples = beta1_samples / beta2_samples
    ratio_32_samples = beta3_samples / beta2_samples

    beta_estimates = accepted_params.mean(axis=0)
    beta_errors = np.abs(beta_estimates - np.array([beta1_true, beta2_true, beta3_true]))

    ratio_12_true = beta1_true / beta2_true
    ratio_32_true = beta3_true / beta2_true

    # Оценки отношений считаются как средние по отношениям принятых параметров.
    # Это удобнее для анализа апостериорного распределения самих отношений.
    ratio_12_est = np.mean(ratio_12_samples)
    ratio_32_est = np.mean(ratio_32_samples)

    ratio_12_error = abs(ratio_12_est - ratio_12_true)
    ratio_32_error = abs(ratio_32_est - ratio_32_true)

    return {
        "epsilon": epsilon,
        "accepted_count": len(accepted_params),
        "acceptance_rate": len(accepted_params) / len(all_params),
        "accepted_params": accepted_params,
        "beta1_est": beta_estimates[0],
        "beta2_est": beta_estimates[1],
        "beta3_est": beta_estimates[2],
        "beta1_error": beta_errors[0],
        "beta2_error": beta_errors[1],
        "beta3_error": beta_errors[2],
        "ratio_12_samples": ratio_12_samples,
        "ratio_32_samples": ratio_32_samples,
        "ratio_12_true": ratio_12_true,
        "ratio_32_true": ratio_32_true,
        "ratio_12_est": ratio_12_est,
        "ratio_32_est": ratio_32_est,
        "ratio_12_error": ratio_12_error,
        "ratio_32_error": ratio_32_error,
    }


def print_results(result):
    print("\nRestoration of parameter ratios")
    print("-" * 100)
    print(
        "epsilon | accepted | acc_rate | "
        "beta1_est | beta2_est | beta3_est | "
        "ratio12_true | ratio12_est | ratio12_err | "
        "ratio32_true | ratio32_est | ratio32_err"
    )
    print("-" * 100)
    print(
        f"{result['epsilon']:<7.4f} | "
        f"{result['accepted_count']:<8d} | "
        f"{result['acceptance_rate']:<8.3f} | "
        f"{result['beta1_est']:<9.4f} | "
        f"{result['beta2_est']:<9.4f} | "
        f"{result['beta3_est']:<9.4f} | "
        f"{result['ratio_12_true']:<12.4f} | "
        f"{result['ratio_12_est']:<11.4f} | "
        f"{result['ratio_12_error']:<11.4f} | "
        f"{result['ratio_32_true']:<12.4f} | "
        f"{result['ratio_32_est']:<11.4f} | "
        f"{result['ratio_32_error']:<11.4f}"
    )

    print("\nParameter estimates")
    print("-" * 70)
    print("parameter | true | estimate | error")
    print("-" * 70)
    print(f"beta1     | 0.2500 | {result['beta1_est']:.4f} | {result['beta1_error']:.4f}")
    print(f"beta2     | 0.0800 | {result['beta2_est']:.4f} | {result['beta2_error']:.4f}")
    print(f"beta3     | 0.2200 | {result['beta3_est']:.4f} | {result['beta3_error']:.4f}")

    print("\nRatio estimates")
    print("-" * 70)
    print("ratio | true | estimate | error")
    print("-" * 70)
    print(
        f"beta1/beta2 | {result['ratio_12_true']:.4f} | "
        f"{result['ratio_12_est']:.4f} | {result['ratio_12_error']:.4f}"
    )
    print(
        f"beta3/beta2 | {result['ratio_32_true']:.4f} | "
        f"{result['ratio_32_est']:.4f} | {result['ratio_32_error']:.4f}"
    )

def plot_ratio_histograms(result):
    ratio_12_samples = result["ratio_12_samples"]
    ratio_32_samples = result["ratio_32_samples"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(ratio_12_samples, bins=25)
    axes[0].axvline(
        result["ratio_12_true"],
        linestyle="--",
        label="Истинное значение",
    )
    axes[0].axvline(
        result["ratio_12_est"],
        linestyle=":",
        label="Оценка",
    )
    axes[0].set_title(r"$\beta_1/\beta_2$")
    axes[0].set_xlabel(r"$\beta_1/\beta_2$")
    axes[0].set_ylabel("Частота")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].hist(ratio_32_samples, bins=25)
    axes[1].axvline(
        result["ratio_32_true"],
        linestyle="--",
        label="Истинное значение",
    )
    axes[1].axvline(
        result["ratio_32_est"],
        linestyle=":",
        label="Оценка",
    )
    axes[1].set_title(r"$\beta_3/\beta_2$")
    axes[1].set_xlabel(r"$\beta_3/\beta_2$")
    axes[1].set_ylabel("Частота")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle("Апостериорные распределения отношений параметров", fontsize=14)
    plt.tight_layout()
    plt.savefig("parameter_ratio_posteriors.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_ratio_estimates(result):
    ratio_names = [r"$\beta_1/\beta_2$", r"$\beta_3/\beta_2$"]
    true_values = [result["ratio_12_true"], result["ratio_32_true"]]
    estimated_values = [result["ratio_12_est"], result["ratio_32_est"]]

    x = np.arange(len(ratio_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, true_values, width, label="Истинное значение")
    plt.bar(x + width / 2, estimated_values, width, label="Оценка")

    plt.xticks(x, ratio_names)
    plt.ylabel("Значение отношения")
    plt.title("Истинные и восстановленные отношения параметров")
    plt.legend()
    plt.grid(axis="y")

    plt.savefig("parameter_ratio_estimates.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_parameter_histograms(result, beta1_true, beta2_true, beta3_true):
    accepted_params = result["accepted_params"]

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

    fig.suptitle("Апостериорные распределения параметров при выбранных настройках", fontsize=14)
    plt.tight_layout()
    plt.savefig("selected_settings_parameter_posteriors.png", dpi=300, bbox_inches="tight")
    plt.show()


beta1_true = 0.25
beta2_true = 0.08
beta3_true = 0.22
gamma_true = 0.1

N = 10_000
I0 = 10
R_initial = 0

T = 140
t1 = 20
t2 = 60
num_points = 141

# Рабочие настройки, выбранные на предыдущих этапах.
sampling_rate = 0.8
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

all_params, all_distances = run_abc(
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

result = analyze_ratios(
    all_params=all_params,
    all_distances=all_distances,
    acceptance_quantile=acceptance_quantile,
    beta1_true=beta1_true,
    beta2_true=beta2_true,
    beta3_true=beta3_true,
)

print_results(result)

plot_parameter_histograms(result, beta1_true, beta2_true, beta3_true)
plot_ratio_histograms(result)
plot_ratio_estimates(result)
