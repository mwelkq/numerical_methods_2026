import csv
import numpy as np
import matplotlib.pyplot as plt
import os


def create_csv(filename):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['n', 't'])
            writer.writerow([100, 120])
            writer.writerow([200, 110])
            writer.writerow([400, 90])
            writer.writerow([800, 65])
            writer.writerow([1600, 40])

        print("CSV файл створено")


# Зчитування даних
def read_data(filename):
    x = []
    y = []

    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))

    return np.array(x), np.array(y)


# Таблиця розділених різниць
def divided_differences(x, y):
    n = len(x)

    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


# Поліном Ньютона
def newton(x, table, value):
    n = len(x)

    result = table[0][0]
    product = 1

    for i in range(1, n):
        product *= value - x[i - 1]
        result += table[0][i] * product

    return result


# Факторіальний поліном
def factorial_poly(x, y, value):
    h = x[1] - x[0]
    s = (value - x[0]) / h

    diff = np.zeros((len(y), len(y)))
    diff[:, 0] = y

    for j in range(1, len(y)):
        for i in range(len(y) - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    result = y[0]
    term = 1

    for i in range(1, len(y)):
        term *= (s - (i - 1)) / i
        result += term * diff[0][i]

    return result


# Поліном Лагранжа
def lagrange(x, y, value):
    n = len(x)

    result = 0

    for i in range(n):
        term = y[i]

        for j in range(n):
            if j != i:
                term *= (value - x[j]) / (x[i] - x[j])

        result += term

    return result


filename = "data.csv"

create_csv(filename)

x, y = read_data(filename)

print("x:", x)
print("y:", y)

# Таблиця різниць
table = divided_differences(x, y)

print("\nТаблиця розділених різниць")
print(table)

# Прогноз FPS для 1000
fps_newton = newton(x, table, 1000)
fps_fact = factorial_poly(x, y, 1000)
fps_lagrange = lagrange(x, y, 1000)

print("\nFPS для 1000 об'єктів")
print("Ньютон:", fps_newton)
print("Факторіальний:", fps_fact)
print("Лагранж:", fps_lagrange)

# Мінімальна кількість об'єктів
objects = np.linspace(100, 2000, 500)
fps_values = [newton(x, table, i) for i in objects]

limit = None

for obj, fps in zip(objects, fps_values):
    if fps < 60:
        limit = obj
        break

print("\nFPS падає нижче 60 приблизно при:", limit, "об'єктах")

# Графік FPS(n)
plt.figure()
plt.scatter(x, y, label="Експериментальні дані")
plt.plot(objects, fps_values, label="Інтерполяція Ньютона")
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.title("FPS(n)")
plt.legend()
plt.grid()
plt.show()

# Дослідження кількості вузлів
nodes = [5, 10, 20]
errors = []

for n in nodes:
    xs = np.linspace(100, 1600, n)
    ys = [lagrange(x, y, i) for i in xs]
    real = [newton(x, table, i) for i in xs]

    error = np.mean(np.abs(np.array(real) - np.array(ys)))
    errors.append(error)

print("\nПохибки для вузлів")

for n, e in zip(nodes, errors):
    print(n, "вузлів -> похибка:", e)

# Графік похибок
plt.figure()
plt.plot(nodes, errors, marker='o')
plt.xlabel("Кількість вузлів")
plt.ylabel("Середня похибка")
plt.title("Залежність похибки від кількості вузлів")
plt.grid()
plt.show()