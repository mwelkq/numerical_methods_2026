import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. Отримання висот з API
# URL із GPS-координатами маршруту
url = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|"
    "48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|"
    "48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|"
    "48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|"
    "48.160250,24.500106"
)

# HTTP-запит до сервера
response = requests.get(url, timeout=30)
response.raise_for_status()
data = response.json()
results = data["results"]

n_all = len(results)
print("Кількість вузлів:", n_all)

# Вивід таблиці координат і висот
print("\n№ | Latitude | Longitude | Elevation (m)")
for i, p in enumerate(results):
    print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")

# Запис таблиці у файл
with open("tabulation_nodes.txt", "w", encoding="utf-8") as f:
    f.write("№ | Latitude | Longitude | Elevation (m)\n")
    for i, p in enumerate(results):
        f.write(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}\n")

# 2. Кумулятивна відстань

# Формула гаверсина для відстані між двома GPS-точками
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0  # радіус Землі (м)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Масиви координат і висот
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations_all = np.array([p["elevation"] for p in results], dtype=float)

# Кумулятивна відстань вздовж маршруту
distances_all = [0.0]
for i in range(1, n_all):
    d = haversine(*coords[i - 1], *coords[i])
    distances_all.append(distances_all[-1] + d)
distances_all = np.array(distances_all, dtype=float)

print("\nЗагальна довжина маршруту (м):", distances_all[-1])

# Запис табуляції (відстань–висота)
with open("tabulation_distance_elevation.txt", "w", encoding="utf-8") as f:
    f.write("№ | Distance (m) | Elevation (m)\n")
    for i in range(n_all):
        f.write(f"{i:2d} | {distances_all[i]:10.2f} | {elevations_all[i]:8.2f}\n")


# 3. Натуральний кубічний сплайн

def cubic_spline_natural(x, y):
    # x – вузли, y – значення у вузлах
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    h = np.diff(x)

    # Коефіцієнти тридіагональної системи
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    # Натуральні граничні умови
    B[0] = 1.0
    B[-1] = 1.0

    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 6 * ((y[i + 1] - y[i]) / h[i] -
                    (y[i] - y[i - 1]) / h[i - 1])

    # Метод прогонки
    alpha = np.zeros(n)
    beta = np.zeros(n)

    for i in range(1, n - 1):
        denom = B[i] - A[i] * alpha[i - 1]
        alpha[i] = C[i] / denom
        beta[i] = (D[i] - A[i] * beta[i - 1]) / denom

    M = np.zeros(n)
    for i in range(n - 2, 0, -1):
        M[i] = beta[i] - alpha[i] * M[i + 1]

    # Коефіцієнти сплайна
    a = y[:-1]
    b = np.zeros(n - 1)
    c = M[:-1] / 2
    d = np.zeros(n - 1)

    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
        d[i] = (M[i + 1] - M[i]) / (6 * h[i])

    return a, b, c, d, x, alpha, beta


# Обчислення значень сплайна
def spline_eval(xx, a, b, c, d, x_nodes):
    yy = np.zeros_like(xx)
    j = 0
    for k in range(len(xx)):
        while j < len(x_nodes) - 2 and xx[k] > x_nodes[j + 1]:
            j += 1
        dx = xx[k] - x_nodes[j]
        yy[k] = a[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3
    return yy


# Перша похідна (градієнт)
def spline_derivative(xx, b, c, d, x_nodes):
    g = np.zeros_like(xx)
    j = 0
    for k in range(len(xx)):
        while j < len(x_nodes) - 2 and xx[k] > x_nodes[j + 1]:
            j += 1
        dx = xx[k] - x_nodes[j]
        g[k] = b[j] + 2*c[j]*dx + 3*d[j]*dx**2
    return g


# 4. Графік усіх вузлів


plt.figure()
plt.plot(distances_all, elevations_all, "o-")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль висоти (усі вузли)")
plt.grid(True)

# 5. Порівняння 10 / 15 / 20 вузлів

# Еталонний сплайн (20 вузлів)
aR, bR, cR, dR, xR, _, _ = cubic_spline_natural(
    distances_all[:20], elevations_all[:20]
)

def run_for_k(k):
    a, b, c, d, x_nodes, alpha, beta = cubic_spline_natural(
        distances_all[:k], elevations_all[:k]
    )

    # Вивід коефіцієнтів прогонки
    print(f"\nПрогонка для k={k}")
    for i in range(1, k - 1):
        print(f"i={i}  alpha={alpha[i]:.6f}  beta={beta[i]:.6f}")

    xx = np.linspace(x_nodes[0], x_nodes[-1], 500)
    yy = spline_eval(xx, a, b, c, d, x_nodes)

    # Графік сплайна
    plt.figure()
    plt.plot(x_nodes, elevations_all[:k], "o", label="Вузли")
    plt.plot(xx, yy, label="Сплайн")
    plt.title(f"Кубічний сплайн (k={k})")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Висота (м)")
    plt.grid(True)
    plt.legend()

    # Похибка відносно еталону
    yy_ref = spline_eval(xx, aR, bR, cR, dR, xR)
    err = np.abs(yy - yy_ref)

    plt.figure()
    plt.plot(xx, err)
    plt.title(f"Похибка (k={k})")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Похибка (м)")
    plt.grid(True)

    # Градієнт
    grad = spline_derivative(xx, b, c, d, x_nodes) * 100

    plt.figure()
    plt.plot(xx, grad)
    plt.title(f"Градієнт (%) k={k}")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Градієнт (%)")
    plt.grid(True)

    return xx, yy, grad


last_xx = last_yy = last_grad = None
for k in (10, 15, 20):
    last_xx, last_yy, last_grad = run_for_k(k)


# 6. Табуляція 500 точок

with open("tabulation_full_500.txt", "w", encoding="utf-8") as f:
    f.write("x(m) | S(x)(m) | grad(%)\n")
    for i in range(len(last_xx)):
        f.write(f"{last_xx[i]:10.3f} | {last_yy[i]:10.3f} | {last_grad[i]:10.6f}\n")


plt.show()