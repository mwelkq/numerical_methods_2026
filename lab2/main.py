import csv
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл '{filename}' не знайдено.")

    x = []
    y = []

    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV-файл порожній або не містить заголовків.")

        headers = [h.strip() for h in reader.fieldnames]
        required_headers = {"Objects", "FPS"}

        if not required_headers.issubset(set(headers)):
            raise ValueError(
                f"У CSV повинні бути стовпці: {required_headers}. Знайдено: {headers}"
            )

        for i, row in enumerate(reader, start=2):
            try:
                xi = float(row["Objects"])
                yi = float(row["FPS"])
            except (ValueError, TypeError):
                raise ValueError(f"Некоректні дані в рядку {i}: {row}")

            if xi <= 0:
                raise ValueError(f"Objects має бути > 0. Помилка в рядку {i}.")
            if yi <= 0:
                raise ValueError(f"FPS має бути > 0. Помилка в рядку {i}.")

            x.append(xi)
            y.append(yi)

    if len(x) < 2:
        raise ValueError("Для інтерполяції потрібно щонайменше 2 вузли.")

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if len(np.unique(x)) != len(x):
        raise ValueError("Значення Objects повинні бути унікальними.")

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    return x, y


def check_monotonic_decreasing(y: np.ndarray) -> bool:
    return np.all(np.diff(y) < 0)


def is_equally_spaced(x: np.ndarray, tol: float = 1e-9) -> bool:
    if len(x) < 3:
        return True
    diffs = np.diff(x)
    return np.all(np.abs(diffs - diffs[0]) < tol)


def divided_differences_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            denominator = x[i + j] - x[i]
            if abs(denominator) < 1e-15:
                raise ZeroDivisionError("Знайдено однакові вузли, ділення на нуль.")
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / denominator

    return table


def get_newton_coefficients(diff_table: np.ndarray) -> np.ndarray:
    return diff_table[0, :]


def newton_polynomial(x_data: np.ndarray, coeffs: np.ndarray, x_value: float) -> float:
    result = coeffs[0]
    product = 1.0

    for i in range(1, len(coeffs)):
        product *= (x_value - x_data[i - 1])
        result += coeffs[i] * product

    return result


def newton_values(x_data: np.ndarray, coeffs: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    return np.array([newton_polynomial(x_data, coeffs, xv) for xv in x_values], dtype=float)


def finite_differences_table(y: np.ndarray) -> np.ndarray:
    n = len(y)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]

    return table


def factorial_forward_interpolation(x_data: np.ndarray, y_data: np.ndarray, x_value: float) -> float:
    if not is_equally_spaced(x_data):
        raise ValueError("Факторіальний многочлен можна застосовувати лише для рівновіддалених вузлів.")

    h = x_data[1] - x_data[0]
    t = (x_value - x_data[0]) / h

    diff_table = finite_differences_table(y_data)
    result = diff_table[0, 0]
    term = 1.0

    for k in range(1, len(x_data)):
        term *= (t - (k - 1)) / k
        result += term * diff_table[0, k]

    return result


def print_divided_differences_table(table: np.ndarray) -> None:
    n = len(table)
    print("\nТаблиця розділених різниць:")
    for i in range(n):
        row_values = []
        for j in range(n - i):
            row_values.append(f"{table[i, j]:12.6f}")
        print(" ".join(row_values))


def find_threshold_objects(
    x_data: np.ndarray,
    coeffs: np.ndarray,
    fps_limit: float,
    left: float,
    right: float,
    eps: float = 1e-6,
    max_iter: int = 200
) -> float:
    f_left = newton_polynomial(x_data, coeffs, left) - fps_limit
    f_right = newton_polynomial(x_data, coeffs, right) - fps_limit

    if f_left < 0:
        raise ValueError("У лівій межі FPS вже менший за поріг.")
    if f_right > 0:
        raise ValueError("У правій межі FPS ще не опустився до порога.")

    for _ in range(max_iter):
        mid = (left + right) / 2
        f_mid = newton_polynomial(x_data, coeffs, mid) - fps_limit

        if abs(f_mid) < eps:
            return mid

        if f_mid > 0:
            left = mid
        else:
            right = mid

    return (left + right) / 2


def leave_one_out_error(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return float("nan")

    errors = []

    for i in range(n):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)

        table = divided_differences_table(x_train, y_train)
        coeffs = get_newton_coefficients(table)

        y_pred = newton_polynomial(x_train, coeffs, x[i])
        errors.append(abs(y_pred - y[i]))

    return float(np.mean(errors))


def main():
    filename = "variant5.csv"

    try:
        x, y = read_data(filename)

        print("Зчитані дані:")
        for xi, yi in zip(x, y):
            print(f"Objects = {xi:.0f}, FPS = {yi:.2f}")

        if not check_monotonic_decreasing(y):
            print("\nПопередження: FPS не є строго спадною функцією.")
        else:
            print("\nПеревірка пройдена: FPS спадає зі збільшенням кількості об'єктів.")

        table = divided_differences_table(x, y)
        coeffs = get_newton_coefficients(table)
        print_divided_differences_table(table)

        target_objects = 1000
        fps_newton = newton_polynomial(x, coeffs, target_objects)
        print(f"\nПрогноз FPS для {target_objects} об'єктів (Ньютон): {fps_newton:.6f}")

        if is_equally_spaced(x):
            fps_factorial = factorial_forward_interpolation(x, y, target_objects)
            print(f"Прогноз FPS для {target_objects} об'єктів (факторіальний): {fps_factorial:.6f}")
        else:
            print("Факторіальний многочлен не обчислювався, бо вузли не є рівновіддаленими.")

        fps_limit = 60.0
        if newton_polynomial(x, coeffs, x[0]) < fps_limit:
            print("Навіть при мінімальній кількості об'єктів FPS < 60.")
        elif newton_polynomial(x, coeffs, x[-1]) > fps_limit:
            print("Навіть при максимальній кількості об'єктів FPS > 60.")
        else:
            threshold = find_threshold_objects(x, coeffs, fps_limit, x[0], x[-1])
            print(f"Гранична кількість об'єктів для FPS >= 60: приблизно {threshold:.2f}")

        cv_error = leave_one_out_error(x, y)
        print(f"\nСередня похибка leave-one-out: {cv_error:.6f}")

        print("\nДослідження кількості вузлів:")
        for count in [5, 10, 20]:
            if count > len(x):
                print(f"{count} вузлів: неможливо дослідити, бо у файлі лише {len(x)} точок.")
            else:
                print(f"{count} вузлів: доступно для дослідження.")

        x_plot = np.linspace(x[0], x[-1], 500)
        y_plot = newton_values(x, coeffs, x_plot)

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, label="Інтерполяційний многочлен Ньютона")
        plt.scatter(x, y, label="Експериментальні точки")
        plt.axhline(60, linestyle="--", label="FPS = 60")
        plt.xlabel("Кількість об'єктів")
        plt.ylabel("FPS")
        plt.title("Прогнозування FPS залежно від кількості об'єктів")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Помилка: {e}")


if __name__ == "__main__":
    main()