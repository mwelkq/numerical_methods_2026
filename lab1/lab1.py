import requests                     # для виконання HTTP-запитів до API
import numpy as np                  # для чисельних обчислень
import matplotlib.pyplot as plt     # для побудови графіків
import os                           # для роботи з файловою системою

# ==========================================================
# 1. Запит до Open-Elevation API
# ==========================================================

url = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

response = requests.get(url)        # надсилання запиту до сервера
data = response.json()              # перетворення відповіді у формат JSON
results = data["results"]           # отримання списку GPS-точок з висотами

n = len(results)                    # кількість точок маршруту

# ==========================================================
# 2. Запис табуляції у файл
# ==========================================================

with open("tabulation.txt", "w", encoding="utf-8") as f:   # створення текстового файлу
    f.write("№ | Latitude | Longitude | Elevation (m)\n") # заголовок таблиці
    for i, p in enumerate(results):                        # перебір усіх точок
        f.write(
            f"{i:2d} | {p['latitude']:.6f} | "
            f"{p['longitude']:.6f} | {p['elevation']:.2f}\n"
        )                                                   # запис координат і висоти

# ==========================================================
# 3. Кумулятивна відстань (формула гаверсинусів)
# ==========================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000                                           # радіус Землі (м)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)      # широти в радіанах
    dphi = np.radians(lat2 - lat1)                        # різниця широт
    dlambda = np.radians(lon2 - lon1)                     # різниця довгот
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )                                                     # формула гаверсинуса
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) # відстань між точками

coords = [(p["latitude"], p["longitude"]) for p in results]   # список координат
elevations = np.array([p["elevation"] for p in results])     # масив висот

distances = [0]                                          # початкова відстань

for i in range(1, n):                                    # обхід маршруту
    d = haversine(*coords[i-1], *coords[i])              # відстань між точками
    distances.append(distances[-1] + d)                  # накопичення довжини

distances = np.array(distances)                          # перетворення у numpy-масив

# ==========================================================
# 4. Метод прогонки (Thomas algorithm)
# ==========================================================

def thomas(a, b, c, d):
    n = len(d)                                           # розмір системи
    c_ = np.zeros(n)                                     # коефіцієнти прямого ходу
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]                                  # початкові значення
    d_[0] = d[0] / b[0]

    for i in range(1, n):                                # прямий хід прогонки
        temp = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / temp if i < n - 1 else 0
        d_[i] = (d[i] - a[i] * d_[i - 1]) / temp

    x = np.zeros(n)                                      # масив розв’язку
    x[-1] = d_[-1]                                       # початок зворотного ходу

    for i in reversed(range(n - 1)):                     # зворотний хід
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x                                             # повернення розв’язку

# ==========================================================
# 5. Кубічний натуральний сплайн
# ==========================================================

def cubic_spline(x, y):
    n = len(x)                                           # кількість вузлів
    h = np.diff(x)                                       # кроки між вузлами

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    alpha = np.zeros(n)                                  # права частина системи

    for i in range(1, n - 1):
        alpha[i] = (
            (3 / h[i]) * (y[i + 1] - y[i])
            - (3 / h[i - 1]) * (y[i] - y[i - 1])
        )

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    B[0] = 1                                             # граничні умови
    B[-1] = 1

    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = alpha[i]

    c = thomas(A, B, C, D)                               # розв’язання системи

    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        a[i] = y[i]

    return a, b, c, d                                   # коефіцієнти сплайна

a, b, c, d_coef = cubic_spline(distances, elevations)   # побудова сплайна

# ==========================================================
# 6. Обчислення значень сплайна
# ==========================================================

def spline_eval(x, x_nodes, a, b, c, d):
    y = np.zeros_like(x)                                 # масив значень
    for i in range(len(x_nodes) - 1):                    # перебір інтервалів
        mask = (x >= x_nodes[i]) & (x <= x_nodes[i + 1])
        dx = x[mask] - x_nodes[i]                        # відстань від вузла
        y[mask] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return y

xx = np.linspace(distances[0], distances[-1], 500)      # щільна сітка
yy = spline_eval(xx, distances, a, b, c, d_coef)        # значення сплайна

# ==========================================================
# 7. Графік профілю маршруту
# ==========================================================

os.makedirs("images", exist_ok=True)                     # папка для графіків

plt.figure(figsize=(8, 5))
plt.plot(distances, elevations, 'o', label="GPS-точки")
plt.plot(xx, yy, label="Кубічний сплайн")
plt.title("Профіль висоти маршруту Заросляк – Говерла")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.savefig("images/graph1.png", dpi=300)
plt.show()

# ==========================================================
# 8. Вплив кількості вузлів
# ==========================================================

plt.figure(figsize=(8, 5))

for k in [10, 15, 20]:                                   # різна кількість вузлів
    idx = np.linspace(0, n - 1, k, dtype=int)
    x_sub = distances[idx]
    y_sub = elevations[idx]

    a1, b1, c1, d1 = cubic_spline(x_sub, y_sub)
    xx_sub = np.linspace(x_sub[0], x_sub[-1], 400)
    yy_sub = spline_eval(xx_sub, x_sub, a1, b1, c1, d1)

    plt.plot(xx_sub, yy_sub, label=f"{k} вузлів")

plt.title("Вплив кількості вузлів на точність")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.savefig("images/graph2.png", dpi=300)
plt.show()

# ==========================================================
# 9. Порівняння та похибка
# ==========================================================

interp_vals = np.interp(xx, distances, elevations)       # лінійна інтерполяція
error = np.abs(interp_vals - yy)                         # абсолютна похибка

plt.figure(figsize=(8, 5))
plt.plot(xx, interp_vals, label="Лінійна інтерполяція")
plt.plot(xx, yy, label="Кубічний сплайн")
plt.plot(xx, error, label="Похибка")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота / Похибка (м)")
plt.legend()
plt.grid(True)
plt.savefig("images/graph3.png", dpi=300)
plt.show()

# ==========================================================
# 10. Характеристики маршруту
# ==========================================================

print("\n===== ХАРАКТЕРИСТИКИ МАРШРУТУ =====")

print("Загальна довжина маршруту (м):", distances[-1])

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
print("Сумарний набір висоти (м):", total_ascent)

total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
print("Сумарний спуск (м):", total_descent)

# ==========================================================
# 11. Аналіз градієнта
# ==========================================================

yy_full = yy
grad_full = np.gradient(yy_full, xx) * 100               # градієнт у %

print("\n===== АНАЛІЗ ГРАДІЄНТА =====")
print("Максимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))

# ==========================================================
# 12. Механічна енергія
# ==========================================================

mass = 80                                                # маса людини (кг)
g = 9.81                                                 # прискорення вільного падіння
energy = mass * g * total_ascent                         # механічна робота

print("\n===== МЕХАНІЧНА ЕНЕРГІЯ =====")
print("Робота (Дж):", energy)
print("Робота (кДж):", energy / 1000)
print("Енергія (ккал):", energy / 4184)