import matplotlib.pyplot as plt

from sir_abc_utils import simulate_sir, generate_observations


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

sampling_rate = 0.5


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


plt.figure(figsize=(10, 6))

plt.plot(t, S, label="S(t) — восприимчивые")
plt.plot(t, I, label="I(t) — инфицированные")
plt.plot(t, R, label="R(t) — выздоровевшие / выбывшие")

plt.axvline(t1, linestyle="--", label="Начало карантина")
plt.axvline(t2, linestyle="--", label="Снятие карантина")

plt.xlabel("Время")
plt.ylabel("Число индивидов")
plt.title("SIR-модель с тремя временными эпохами")
plt.legend()
plt.grid(True)

plt.savefig("SIR-model_3_timelines.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 6))

plt.plot(t, I, label="Истинное I(t)")
plt.plot(
    t,
    I_obs,
    label=f"Наблюдаемое I_obs(t), sampling rate = {sampling_rate}",
)

plt.axvline(t1, linestyle="--", label="Начало карантина")
plt.axvline(t2, linestyle="--", label="Снятие карантина")

plt.xlabel("Время")
plt.ylabel("Число инфицированных")
plt.title("Истинная и наблюдаемая динамика инфицированных")
plt.legend()
plt.grid(True)

plt.savefig("True_and_real_lines.png", dpi=300, bbox_inches="tight")
plt.show()