import csv
import matplotlib.pyplot as plt


def read_csv_data(filename):
    x = []
    y = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["Month"]))
            y.append(float(row["Temp"]))

    return x, y


def form_matrix(x, m):
    n = m + 1
    A = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(len(x)):
                s += x[k] ** (i + j)
            A[i][j] = s

    return A


def form_vector(x, y, m):
    n = m + 1
    b = [0.0 for _ in range(n)]

    for i in range(n):
        s = 0.0
        for k in range(len(x)):
            s += y[k] * (x[k] ** i)
        b[i] = s

    return b


def gauss_solve(A, b):
    n = len(A)

    A = [row[:] for row in A]
    b = b[:]

    for k in range(n):
        max_row = k
        max_value = abs(A[k][k])

        for i in range(k + 1, n):
            if abs(A[i][k]) > max_value:
                max_value = abs(A[i][k])
                max_row = i

        if max_value == 0:
            raise ValueError("Система не має єдиного розв'язку.")

        if max_row != k:
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x_sol = [0.0 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * x_sol[j]

        if A[i][i] == 0:
            raise ValueError("Неможливо виконати зворотний хід методу Гауса.")

        x_sol[i] = (b[i] - s) / A[i][i]

    return x_sol


def polynomial_value(x_value, coef):
    result = 0.0
    for i in range(len(coef)):
        result += coef[i] * (x_value ** i)
    return result


def polynomial(x_values, coef):
    y_poly = []
    for x_value in x_values:
        y_poly.append(polynomial_value(x_value, coef))
    return y_poly


def variance(y_true, y_approx):
    s = 0.0
    n = len(y_true)

    for i in range(n):
        s += (y_true[i] - y_approx[i]) ** 2

    return s / n


def errors(y_true, y_approx):
    err = []
    for i in range(len(y_true)):
        err.append(y_true[i] - y_approx[i])
    return err


def print_polynomial(coef):
    parts = []
    for i, c in enumerate(coef):
        if i == 0:
            parts.append(f"{c:.6f}")
        elif i == 1:
            parts.append(f"{c:+.6f}*x")
        else:
            parts.append(f"{c:+.6f}*x^{i}")
    return " ".join(parts)


def save_results(filename, degrees, variances, optimal_m, coef, x_future, y_future):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТИ ЛАБОРАТОРНОЇ РОБОТИ\n")
        f.write("=" * 50 + "\n\n")

        f.write("Дисперсії для різних степенів полінома:\n")
        for d, v in zip(degrees, variances):
            f.write(f"m = {d}: D = {v:.6f}\n")

        f.write(f"\nОптимальний степінь полінома: m = {optimal_m}\n")
        f.write("Коефіцієнти полінома:\n")
        for i, c in enumerate(coef):
            f.write(f"a{i} = {c:.6f}\n")

        f.write("\nПоліном:\n")
        f.write(print_polynomial(coef) + "\n")

        f.write("\nПрогноз на наступні 3 місяці:\n")
        for xm, ym in zip(x_future, y_future):
            f.write(f"Місяць {int(xm)}: {ym:.6f}\n")


def save_error_table(filename, x, all_errors):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Таблиця похибок апроксимації\n")
        f.write("Month")
        for m in sorted(all_errors.keys()):
            f.write(f"\tm={m}")
        f.write("\n")

        for i in range(len(x)):
            f.write(f"{int(x[i])}")
            for m in sorted(all_errors.keys()):
                f.write(f"\t{all_errors[m][i]:.6f}")
            f.write("\n")


def plot_variances(degrees, variances):
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, variances, marker='o')
    plt.title("Залежність дисперсії від степеня полінома")
    plt.xlabel("Степінь полінома m")
    plt.ylabel("Дисперсія")
    plt.grid(True)
    plt.savefig("variance_plot.png")
    plt.show()


def plot_approximation(x, y, x_future, y_future, coef, optimal_m):
    x_dense = []
    start = min(x)
    end = max(x_future)
    points = 300

    for i in range(points):
        x_dense.append(start + i * (end - start) / (points - 1))

    y_dense = polynomial(x_dense, coef)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label="Фактичні дані")
    plt.plot(x_dense, y_dense, '-', label=f"Апроксимація, m={optimal_m}")
    plt.plot(x_future, y_future, 's--', label="Прогноз на 3 місяці")
    plt.title("Апроксимація температур методом найменших квадратів")
    plt.xlabel("Місяць")
    plt.ylabel("Температура")
    plt.legend()
    plt.grid(True)
    plt.savefig("approximation_plot.png")
    plt.show()


def plot_errors(x, all_errors):
    plt.figure(figsize=(10, 6))

    for m in sorted(all_errors.keys()):
        plt.plot(x, all_errors[m], marker='o', label=f"Похибка m={m}")

    plt.title("Графіки похибки апроксимації")
    plt.xlabel("Місяць")
    plt.ylabel("Похибка")
    plt.legend()
    plt.grid(True)
    plt.savefig("error_plot.png")
    plt.show()


def main():
    x, y = read_csv_data("temperature.csv")

    max_degree = 4
    degrees = []
    variances = []
    all_errors = {}
    coefficients_by_degree = {}

    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b = form_vector(x, y, m)
        coef = gauss_solve(A, b)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        err = errors(y, y_approx)

        degrees.append(m)
        variances.append(var)
        all_errors[m] = err
        coefficients_by_degree[m] = coef

    min_index = variances.index(min(variances))
    optimal_m = degrees[min_index]

    coef = coefficients_by_degree[optimal_m]
    y_approx = polynomial(x, coef)

    x_future = [25, 26, 27]
    y_future = polynomial(x_future, coef)

    print("Дисперсії для різних степенів полінома:")
    for d, v in zip(degrees, variances):
        print(f"m = {d}, D = {v:.6f}")

    print(f"\nОптимальний степінь полінома: m = {optimal_m}")

    print("\nКоефіцієнти полінома:")
    for i, c in enumerate(coef):
        print(f"a{i} = {c:.6f}")

    print("\nШуканий поліном:")
    print(print_polynomial(coef))

    print("\nПрогноз на наступні 3 місяці:")
    for xm, ym in zip(x_future, y_future):
        print(f"Місяць {xm}: {ym:.6f}")

    save_results("results.txt", degrees, variances, optimal_m, coef, x_future, y_future)
    save_error_table("errors_table.txt", x, all_errors)

    plot_variances(degrees, variances)
    plot_approximation(x, y, x_future, y_future, coef, optimal_m)
    plot_errors(x, all_errors)


if __name__ == "__main__":
    main()