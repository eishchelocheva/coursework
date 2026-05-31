"""
08_final_selected_settings.py

Итоговая проверка восстановления параметров SIR-модели методом rejection ABC
при выбранных настройках алгоритма.

Выбранные настройки:
- sampling_rate p = 0.8;
- n_simulations = 3000;
- доля принятия q = 0.02;
- сводные статистики peak_load_final:
  1) максимум I(t),
  2) суммарная эпидемическая нагрузка sum I(t),
  3) значение I(t) в конце периода.

Скрипт сохраняет графики:
- final_selected_settings_posteriors.png
- final_selected_settings_estimates.png
- final_selected_settings_curves.png
"""

import numpy as np
import matplotlib.pyplot as plt

from sir_abc_utils import (
    simulate_sir,
    generate_observations,
)


np.random.seed(42)


# -----------------------------
# 1. Summary statistics and distance
# -----------------------------

def summary_peak_load_final(t, I):
    """
    Рабочий набор сводных статистик peak_load_final:
    1) максимум I(t),
    2) суммарная эпидемическая нагрузка sum I(t),
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
    Нормировка нужна, потому что статистики имеют разные масштабы.
    """
    scale = np.maximum(np.abs(stats_obs), 1.0)
    normalized_diff = (stats_sim - stats_obs) / scale

    return np.sqrt(np.sum(normalized_diff ** 2))


# -----------------------------
# 2. ABC simulation
# -----------------------------

def run_abc_final(
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
    Запускает rejection ABC при выбранных настройках.
    Возвращает все параметры-кандидаты и расстояния до наблюдаемых данных.
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

        # Симулированные данные приводятся к той же модели наблюдения,
        # что и наблюдаемые данные.
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


# -----------------------------
# 3. Analysis helpers
# -----------------------------

def summarize_final_results(
    all_params,
    all_distances,
    acceptance_quantile,
    beta_true,
    n_simulations,
):
    """
    Вычисляет epsilon, принятые параметры, оценки и ошибки.
    """
    epsilon = np.quantile(all_distances, acceptance_quantile)
    accepted_mask = all_distances <= epsilon
    accepted_params = all_params[accepted_mask]

    beta_estimates = accepted_params.mean(axis=0)
    beta_errors = np.abs(beta_estimates - beta_true)

    return {
        "epsilon": epsilon,
        "accepted_count": len(accepted_params),
        "acceptance_rate": len(accepted_params) / n_simulations,
        "beta1_est": beta_estimates[0],
        "beta2_est": beta_estimates[1],
        "beta3_est": beta_estimates[2],
        "beta1_error": beta_errors[0],
        "beta2_error": beta_errors[1],
        "beta3_error": beta_errors[2],
        "accepted_params": accepted_params,
    }


def print_final_results(result, sampling_rate, n_simulations, acceptance_quantile):
    """
    Печатает итоговую таблицу для отчёта.
    """
    print("\nFinal selected settings")
    print("-" * 100)
    print(f"sampling_rate p      = {sampling_rate}")
    print(f"n_simulations        = {n_simulations}")
    print(f"acceptance_quantile  = {acceptance_quantile}")
    print("summary statistics   = peak_load_final")
    print("-" * 100)
    print(
        "epsilon | accepted | acc_rate | "
        "beta1_est | beta2_est | beta3_est | "
        "beta1_err | beta2_err | beta3_err"
    )
    print("-" * 100)
    print(
        f"{result['epsilon']:<7.4f} | "
        f"{result['accepted_count']:<8d} | "
        f"{result['acceptance_rate']:<8.3f} | "
        f"{result['beta1_est']:<9.4f} | "
        f"{result['beta2_est']:<9.4f} | "
        f"{result['beta3_est']:<9.4f} | "
        f"{result['beta1_error']:<9.4f} | "
        f"{result['beta2_error']:<9.4f} | "
        f"{result['beta3_error']:<9.4f}"
    )


# -----------------------------
# 4. Plots
# -----------------------------

def plot_final_posteriors(result, beta1_true, beta2_true, beta3_true):
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

    plt.savefig("final_selected_settings_posteriors.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_final_estimates(result, beta1_true, beta2_true, beta3_true):
    names = [r"$\beta_1$", r"$\beta_2$", r"$\beta_3$"]
    true_values = [beta1_true, beta2_true, beta3_true]
    estimated_values = [
        result["beta1_est"],
        result["beta2_est"],
        result["beta3_est"],
    ]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, true_values, width, label="Истинное значение")
    plt.bar(x + width / 2, estimated_values, width, label="Оценка")

    plt.xticks(x, names)
    plt.ylabel("Значение параметра")
    plt.title("Истинные и восстановленные значения параметров")
    plt.legend()
    plt.grid(True, axis="y")

    plt.savefig("final_selected_settings_estimates.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_final_curves(
    t,
    I_true,
    I_obs,
    beta_estimates,
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
    Сравнивает истинную кривую, наблюдаемую кривую и кривую,
    полученную при восстановленных параметрах.
    """
    t_est, _, I_est, _ = simulate_sir(
        beta1=beta_estimates[0],
        beta2=beta_estimates[1],
        beta3=beta_estimates[2],
        gamma=gamma_true,
        N=N,
        I0=I0,
        R_initial=R_initial,
        T=T,
        t1=t1,
        t2=t2,
        num_points=num_points,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(t, I_true, label="Истинное I(t)")
    plt.plot(t, I_obs, label="Наблюдаемое I_obs(t)")
    plt.plot(t_est, I_est, label="I(t) при восстановленных параметрах")

    plt.axvline(t1, linestyle="--", label="Начало карантина")
    plt.axvline(t2, linestyle="--", label="Снятие карантина")

    plt.xlabel("Время")
    plt.ylabel("Число инфицированных")
    plt.title("Сравнение истинной, наблюдаемой и восстановленной динамики")
    plt.legend()
    plt.grid(True)

    plt.savefig("final_selected_settings_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# 5. Main run
# -----------------------------

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

# Выбранные рабочие настройки.
sampling_rate = 0.8
n_simulations = 3000
acceptance_quantile = 0.02


# Генерируем истинную траекторию.
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

# Генерируем наблюдаемые данные при p = 0.8.
I_obs = generate_observations(
    I,
    sampling_rate=sampling_rate,
    random_seed=42,
)

# Запускаем ABC.
all_params, all_distances = run_abc_final(
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

result = summarize_final_results(
    all_params=all_params,
    all_distances=all_distances,
    acceptance_quantile=acceptance_quantile,
    beta_true=beta_true,
    n_simulations=n_simulations,
)

print_final_results(
    result=result,
    sampling_rate=sampling_rate,
    n_simulations=n_simulations,
    acceptance_quantile=acceptance_quantile,
)

beta_estimates = np.array([
    result["beta1_est"],
    result["beta2_est"],
    result["beta3_est"],
])

plot_final_posteriors(result, beta1_true, beta2_true, beta3_true)
plot_final_estimates(result, beta1_true, beta2_true, beta3_true)
plot_final_curves(
    t=t,
    I_true=I,
    I_obs=I_obs,
    beta_estimates=beta_estimates,
    gamma_true=gamma_true,
    N=N,
    I0=I0,
    R_initial=R_initial,
    T=T,
    t1=t1,
    t2=t2,
    num_points=num_points,
)
