import numpy as np
from scipy.integrate import solve_ivp


def beta_t(t, beta1, beta2, beta3, t1, t2):
    if t < t1:
        return beta1
    elif t < t2:
        return beta2
    else:
        return beta3


def sir_rhs(t, y, beta1, beta2, beta3, gamma, N, t1, t2):
    S, I, R = y
    beta = beta_t(t, beta1, beta2, beta3, t1, t2)

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]


def simulate_sir(
    beta1=0.25,
    beta2=0.08,
    beta3=0.22,
    gamma=0.1,
    N=10_000,
    I0=10,
    R_initial=0,
    T=140,
    t1=20,
    t2=60,
    num_points=141,
):
    S0 = N - I0 - R_initial
    y0 = [S0, I0, R_initial]

    t_eval = np.linspace(0, T, num_points)

    solution = solve_ivp(
        fun=sir_rhs,
        t_span=(0, T),
        y0=y0,
        t_eval=t_eval,
        args=(beta1, beta2, beta3, gamma, N, t1, t2),
        rtol=1e-8,
        atol=1e-8,
    )

    if not solution.success:
        raise RuntimeError("SIR simulation failed")

    S, I, R = solution.y
    return solution.t, S, I, R


def generate_observations(I, sampling_rate=1.0, random_seed=None):
    rng = np.random.default_rng(random_seed)
    I_int = np.round(I).astype(int)

    return rng.binomial(
        n=I_int,
        p=sampling_rate,
    )


def compute_summary_statistics(t, I):
    peak_size = np.max(I)
    peak_time = t[np.argmax(I)]
    total_infected_load = np.sum(I)

    return np.array([
        peak_size,
        peak_time,
        total_infected_load,
    ])


def distance_summary_stats(stats_sim, stats_obs):
    scale = np.maximum(np.abs(stats_obs), 1.0)
    normalized_diff = (stats_sim - stats_obs) / scale

    return np.sqrt(np.sum(normalized_diff ** 2))


def run_abc_simulations(
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
    all_params = []
    all_distances = []

    t_obs = np.linspace(0, T, num_points)
    stats_obs = compute_summary_statistics(t_obs, I_obs)

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

        stats_sim = compute_summary_statistics(t_sim, I_sim_obs)
        distance = distance_summary_stats(stats_sim, stats_obs)

        all_params.append([
            beta1_candidate,
            beta2_candidate,
            beta3_candidate,
        ])
        all_distances.append(distance)

    return np.array(all_params), np.array(all_distances)


def analyze_thresholds(all_params, all_distances, beta_true, quantiles, n_simulations):
    results = []

    for q in quantiles:
        epsilon = np.quantile(all_distances, q)
        accepted_mask = all_distances <= epsilon
        accepted_params = all_params[accepted_mask]

        beta_estimates = accepted_params.mean(axis=0)
        errors = np.abs(beta_estimates - beta_true)

        results.append({
            "quantile": q,
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
        })

    return results