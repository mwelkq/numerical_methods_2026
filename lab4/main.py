import numpy as np
import matplotlib.pyplot as plt

# 1. Вихідна функція та її аналітична похідна [cite: 136, 152]
def M(t):# створення функції яка описує вологість ґрунту в часі t
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t) #експонента висихання та коливання вологості

def dM_analytical(t): # аналітична похідна для перевірки точності
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Точка обчислення [cite: 153]
t0 = 1.0 #момент часу для якого ми шукаємо швидкість зміни вологості
exact_val = dM_analytical(t0) #

# 2. Дослідження залежності похибки від кроку h [cite: 190]
h_values = np.logspace(-20, 1, 100) #
errors = [] #

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

for h in h_values:
    approx = central_diff(M, t0, h)
    errors.append(abs(approx - exact_val))

# Пошук оптимального h0 [cite: 190, 191]
min_error_idx = np.argmin(errors)
h0 = h_values[min_error_idx]
r0 = errors[min_error_idx]

# 3. Розрахунки для методів уточнення (h = 10^-3) [cite: 192]
h_base = 1e-3
d_h = central_diff(M, t0, h_base)         # y'(h)
d_2h = central_diff(M, t0, 2 * h_base)    # y'(2h)
d_4h = central_diff(M, t0, 4 * h_base)    # y'(4h)

# 6. Метод Рунге-Ромберга [cite: 196]
y_rr = d_h + (d_h - d_2h) / 3
r_rr = abs(y_rr - exact_val)

# 7. Метод Ейткена [cite: 207]
numerator = (d_2h**2) - (d_4h * d_h)
denominator = 2 * d_2h - (d_4h + d_h)
y_aitken = numerator / denominator

# Порядок точності p [cite: 207]
p_val = np.log(abs((d_4h - d_2h) / (d_2h - d_h))) / np.log(2)
r_aitken = abs(y_aitken - exact_val)

# --- Вивід результатів у термінал ---
print(f"--- Результати Лабораторної роботи №4 ---")
print(f"Точне значення похідної в t0={t0}: {exact_val:.10f} [cite: 155]")
print("-" * 40)
print(f"Оптимальний крок h0: {h0:.2e} [cite: 190]")
print(f"Мінімальна похибка R0: {r0:.2e} [cite: 191]")
print("-" * 40)
print(f"Метод Рунге-Ромберга (h={h_base}):")
print(f"  Уточнене значення: {y_rr:.10f} [cite: 196]")
print(f"  Похибка R2: {r_rr:.2e} [cite: 197]")
print("-" * 40)
print(f"Метод Ейткена:")
print(f"  Уточнене значення: {y_aitken:.10f} [cite: 207]")
print(f"  Оцінка порядку точності p: {p_val:.2f} [cite: 178, 207]")
print(f"  Похибка R3: {r_aitken:.2e} [cite: 208]")

# --- Побудова графіків ---

# Графік 1: Залежність похибки від кроку h (Log-Log) [cite: 190]
plt.figure(figsize=(10, 5))
plt.loglog(h_values, errors, label='Похибка $|y\'_{approx} - y\'_{exact}|$')
plt.axvline(h0, color='r', linestyle='--', label=f'Оптимальне h={h0:.1e}')
plt.title("Залежність похибки чисельного диференціювання від кроку h")
plt.xlabel("Крок h")
plt.ylabel("Абсолютна похибка")
plt.grid(True, which="both", ls="-")
plt.legend()

# Графік 2: Модель вологості ґрунту [cite: 138]
plt.figure(figsize=(10, 5))
t_plot = np.linspace(0, 20, 400)
plt.plot(t_plot, M(t_plot), color='teal', label='M(t) Вологість')
plt.title("Модель вологості ґрунту M(t) [cite: 138]")
plt.xlabel("Час t")
plt.ylabel("M(t)")
plt.grid(True)
plt.legend()

plt.show()