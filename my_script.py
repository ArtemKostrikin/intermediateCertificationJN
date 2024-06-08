from scipy.optimize import fsolve

def f(x):
    return -12 * x**4 * np.sin(np.cos(x)) - 18 * x**3 + 5 * x**2 + 10 * x - 30

roots = fsolve(f, [-10, 10])
print("Roots found at x =", roots)





import sympy as sp
import matplotlib.pyplot as plt
import numpy as np


x = sp.Symbol('x')
f = -12*x**4*sp.sin(sp.cos(x)) - 18*x**3 + 5*x**2 + 10*x - 30
 
f_prime = sp.diff(f, x)

f_prime_func = sp.lambdify(x, f_prime, 'numpy')

x_values = np.linspace(-10, 10, 400)
y_values = f_prime_func(x_values)

plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('График производной функции f(x)')
plt.grid(True)
plt.show()





import sympy as sp
import numpy as np
from scipy.optimize import minimize_scalar

x = sp.Symbol('x')
f = -12*x**4*sp.sin(sp.cos(x)) - 18*x**3 + 5*x**2 + 10*x - 30

f_value = sp.lambdify(x, f)

f_double_prime = sp.diff(sp.diff(f, x), x)
f_double_prime_value = sp.lambdify(x, f_double_prime)

x_initial_guesses = np.linspace(-10, 10, 100)
critical_points = set()
for x_guess in x_initial_guesses:
    res = minimize_scalar(f_value, method='bounded', bounds=(-10, 10), tol=1e-6)
    if res.success:
        critical_points.add(round(res.x, 6))  # Округляем до 6 знаков после запятой

for point in critical_points:
    f_double_prime_val = f_double_prime_value(point)
    if f_double_prime_val > 0:
        vertex_type = "минимум"
    elif f_double_prime_val < 0:
        vertex_type = "максимум"
    else:
        vertex_type = "неопределен"
    
    print("Точка экстремума:", point)
    print("Тип экстремума:", vertex_type)
    print("Значение функции в этой точке:", f_value(point))











