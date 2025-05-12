import numpy as np
import matplotlib.pyplot as plt
import copy
import math


x_train = np.array([4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
y_train = np.array([2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0])

print("Type of x_train:", type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

print("Type of y_train:", type(y_train))
print("First five elements of y_train are:\n", y_train[:5])

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0

    sum_cost = 0
    for i in range(m):
        fwb = (w * x[i]) + b
        cost = (fwb - y[i]) ** 2
        sum_cost += cost

    total_cost = (1 / (2 * m)) * sum_cost
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        fwb = (w * x[i]) + b
        dj_dw_i = (fwb - y[i]) * x[i]
        dj_db_i = (fwb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    J_history = []
    w_history = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

initial_w = 0
initial_b = 0

iterations = 1000
alpha = 0.01

w_final, b_final, J_hist, w_hist = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print(f"\nFinal parameters: w = {w_final:.2f}, b = {b_final:.2f}")
predicted = w_final * x_train + b_final

plt.plot(x_train, predicted, c="b", label="Prediction")
plt.scatter(x_train, y_train, marker='x', c='r', label="Actual Data")
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.legend()
plt.show()
