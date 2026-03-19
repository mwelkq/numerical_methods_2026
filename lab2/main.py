import csv
import math
import os
import matplotlib.pyplot as plt


CSV_FILE = "variant5.csv"
RESULT_FILE = "results.txt"
PREDICT_X = 1000
FPS_LIMIT = 60


def read_data(filename):
    x = []
    y = []

    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["Objects"]))
            y.append(float(row["FPS"]))

    return x, y


def divided_differences(x, y):
    n = len(x)
    table = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


def newton_interpolation(x, y, value):
    table = divided_differences(x, y)
    n = len(x)

    result = table[0][0]
    product = 1.0

    for j in range(1, n):
        product *= (value - x[j - 1])
        result += table[0][j] * product

    return result


def finite_differences(y):
    n = len(y)
    table = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    return table


def is_equally_spaced(x, eps=1e-9):
    if len(x) < 2:
        return False

    h = x[1] - x[0]
    for i in range(2, len(x)):
        if abs((x[i] - x[i - 1]) - h) > eps:
            return False

    return True


def factorial_term(t, k):
    result = 1.0
    for i in range(k):
        result *= (t - i)
    return result


def factorial_interpolation(x, y, value):
    if not is_equally_spaced(x):
        raise ValueError("Факторіальний многочлен напряму застосовується тільки для рівновіддалених вузлів.")

    h = x[1] - x[0]
    t = (value - x[0]) / h

    diff_table = finite_differences(y)

    result = 0.0
    for k in range(len(x)):
        result += diff_table[0][k] * factorial_term(t, k) / math.factorial(k)

    return result


def factorial_interpolation_log2(x, y, value):
    """
    Для варіанта 5 вузли 100, 200, 400, 800, 1600 не є рівновіддаленими по x,
    але є рівновіддаленими після заміни z = log2(x / x0).
    """
    if value <= 0 or any(v <= 0 for v in x):
        raise ValueError("Для log2-перетворення всі x повинні бути додатними.")

    x0 = x[0]
    z = [math.log2(xi / x0) for xi in x]
    zv = math.log2(value / x0)

    return factorial_interpolation(z, y, zv)


def print_divided_differences_table(x, table):
    print("Таблиця розділених різниць:")
    for i in range(len(x)):
        row = [f"x[{i}] = {x[i]:.2f}"]
        for j in range(len(x) - i):
            row.append(f"{table[i][j]:.6f}")
        print(" | ".join(row))


def find_threshold_x(x_min, x_max, func, limit):
    """
    Шукає приблизне максимальне x, при якому FPS >= limit
    """
    left = x_min
    right = x_max
    step = 1.0

    best_x = None
    current = left

    while current <= right:
        if func(current) >= limit:
            best_x = current
        current += step

    return best_x


def lagrange_interpolation(x, y, value):
    n = len(x)
    result = 0.0

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (value - x[j]) / (x[i] - x[j])
        result += term

    return result


def mean_abs_error(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def research_errors(x, y):
    """
    За методичкою треба n = 5, 10, 20.
    Якщо у файлі лише 5 вузлів, реально можна дослідити тільки 5.
    """
    counts = [5, 10, 20]
    available = [c for c in counts if c <= len(x)]

    if not available:
        available = [len(x)]

    results = []

    for n in available:
        x_n = x[:n]
        y_n = y[:n]

        test_x = []
        newton_pred = []
        factorial_pred = []

        for xi in x_n:
            test_x.append(xi)
            newton_pred.append(newton_interpolation(x_n, y_n, xi))

            try:
                factorial_pred.append(factorial_interpolation_log2(x_n, y_n, xi))
            except Exception:
                factorial_pred.append(None)

        newton_error = mean_abs_error(y_n, newton_pred)

        if all(v is not None for v in factorial_pred):
            factorial_error = mean_abs_error(y_n, factorial_pred)
        else:
            factorial_error = None

        results.append((n, newton_error, factorial_error))

    return results


def save_results(filename, text):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    if not os.path.exists(CSV_FILE):
        print(f"Файл {CSV_FILE} не знайдено.")
        return

    x, y = read_data(CSV_FILE)

    dd_table = divided_differences(x, y)
    print_divided_differences_table(x, dd_table)

    newton_value = newton_interpolation(x, y, PREDICT_X)
    lagrange_value = lagrange_interpolation(x, y, PREDICT_X)

    try:
        factorial_value = factorial_interpolation_log2(x, y, PREDICT_X)
        factorial_text = f"{factorial_value:.6f}"
    except Exception as e:
        factorial_value = None
        factorial_text = f"не обчислено ({e})"

    fps_function = lambda value: newton_interpolation(x, y, value)
    max_objects_for_60fps = find_threshold_x(min(x), max(x), fps_function, FPS_LIMIT)

    research = research_errors(x, y)

    result_text = []
    result_text.append("ЛАБОРАТОРНА РОБОТА №2")
    result_text.append("Варіант 5. Оптимізація ігрового рушія. Прогнозування FPS\n")

    result_text.append("Вхідні дані:")
    for xi, yi in zip(x, y):
        result_text.append(f"Objects = {xi:.0f}, FPS = {yi:.2f}")

    result_text.append("\nПрогноз для 1000 об'єктів:")
    result_text.append(f"Ньютон: {newton_value:.6f}")
    result_text.append(f"Факторіальний многочлен: {factorial_text}")
    result_text.append(f"Лагранж: {lagrange_value:.6f}")
    result_text.append(f"\nМаксимальна кількість об'єктів, при якій FPS >= 60: {max_objects_for_60fps:.0f}")

    result_text.append("\nДослідження похибок:")
    for n, err_newton, err_factorial in research:
        result_text.append(f"n = {n}:")
        result_text.append(f"  Похибка Ньютона = {err_newton:.10f}")
        if err_factorial is not None:
            result_text.append(f"  Похибка факторіального многочлена = {err_factorial:.10f}")
        else:
            result_text.append("  Похибка факторіального многочлена = не обчислено")

    save_results(RESULT_FILE, "\n".join(result_text))

    print("\n" + "\n".join(result_text))

    x_plot = []
    y_newton = []
    y_factorial = []

    start = int(min(x))
    end = int(max(x))

    for value in range(start, end + 1, 10):
        x_plot.append(value)
        y_newton.append(newton_interpolation(x, y, value))

        try:
            y_factorial.append(factorial_interpolation_log2(x, y, value))
        except Exception:
            y_factorial.append(None)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Експериментальні точки")
    plt.plot(x_plot, y_newton, label="Інтерполяційний многочлен Ньютона")

    if all(v is not None for v in y_factorial):
        plt.plot(x_plot, y_factorial, label="Факторіальний многочлен")

    plt.axhline(y=60, linestyle="--", label="FPS = 60")
    plt.axvline(x=PREDICT_X, linestyle="--", label="n = 1000")

    plt.xlabel("Objects")
    plt.ylabel("FPS")
    plt.title("FPS(n)")
    plt.grid(True)
    plt.legend()
    plt.show()

    if research:
        nodes = [item[0] for item in research]
        newton_errors = [item[1] for item in research]
        factorial_errors = [item[2] if item[2] is not None else 0 for item in research]

        plt.figure(figsize=(8, 5))
        plt.plot(nodes, newton_errors, marker="o", label="Ньютон")
        plt.plot(nodes, factorial_errors, marker="o", label="Факторіальний многочлен")
        plt.xlabel("Кількість вузлів")
        plt.ylabel("Похибка")
        plt.title("Графік похибок")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()