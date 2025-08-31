import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ANSI escape-послідовності для кольору та скидання кольору
RED = '\033[91m'
GREEN = '\033[92m' # Код для яскраво-зеленого кольору
YELLOW = '\033[93m'  # Код для яскраво-жовтого кольору
BLUE = '\033[94m'  # Код для яскраво-синього кольору
RESET = '\033[0m'

# Визначення функції Сфери
def sphere_function(x):
    """
    Обчислює значення функції Сфери.
    f(x) = sum(xi^2)
    """
    return sum(xi ** 2 for xi in x)

# ---
# Hill Climbing
# ---
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    """
    Алгоритм "Підйом на гору" (Hill Climbing) для мінімізації функції.
    
    Обирає випадкову початкову точку, потім на кожній ітерації шукає 
    кращого сусіда. Якщо кращий сусід знайдений, переходить до нього.
    В іншому випадку, якщо покращення немає, алгоритм зупиняється.
    """
    # Ініціалізація випадкової початкової точки
    solution = [random.uniform(b[0], b[1]) for b in bounds]
    value = func(solution)

    for i in range(iterations):
        # Генеруємо сусідню точку, додаючи невелику випадкову зміну
        neighbor = [
            solution[j] + random.uniform(-0.1, 0.1) 
            for j in range(len(bounds))
        ]
        
        # Обмежуємо сусіда в межах
        for j in range(len(bounds)):
            neighbor[j] = max(bounds[j][0], min(bounds[j][1], neighbor[j]))

        neighbor_value = func(neighbor)

        # Якщо сусід кращий, переходимо до нього
        if neighbor_value < value:
            if abs(value - neighbor_value) < epsilon:
                break
            solution = neighbor
            value = neighbor_value
        else:
            # Зупинка, якщо не знайдено кращого сусіда
            break
            
    return solution, value

# ---
# Random Local Search
# ---
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    """
    Алгоритм "Випадковий локальний пошук" (Random Local Search).
    
    Генерує випадкові сусідні точки та приймає кращу з них.
    Якщо жодна з випадкових спроб не дає покращення, зупиняється.
    """
    # Ініціалізація випадкової початкової точки
    solution = [random.uniform(b[0], b[1]) for b in bounds]
    value = func(solution)
    
    for i in range(iterations):
        # Зберігаємо поточний стан для порівняння
        current_solution = list(solution)
        current_value = value

        # Генеруємо нову випадкову точку в межах
        new_solution = [random.uniform(b[0], b[1]) for b in bounds]
        new_value = func(new_solution)
        
        # Якщо нова точка краща, переходимо до неї
        if new_value < value:
            if abs(value - new_value) < epsilon:
                break
            solution = new_solution
            value = new_value
        else:
            # У цьому випадку, якщо покращення немає,
            # алгоритм продовжує пошук, але не змінює поточне рішення, 
            # якщо воно не покращується
            pass
            
    return solution, value

# ---
# Simulated Annealing
# ---
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    """
    Алгоритм "Імітація відпалу" (Simulated Annealing).
    
    Починає з високої "температури", яка дозволяє приймати гірші рішення, 
    щоб уникнути застрягання в локальних мінімумах. З часом температура
    знижується, і алгоритм стає більш схильним до прийняття кращих рішень.
    """
    # Ініціалізація випадкової початкової точки
    solution = [random.uniform(b[0], b[1]) for b in bounds]
    value = func(solution)
    
    current_temp = temp

    for i in range(iterations):
        # Генеруємо випадкову сусідню точку
        neighbor = [
            solution[j] + random.uniform(-1, 1) * current_temp / temp 
            for j in range(len(bounds))
        ]
        
        # Обмежуємо сусіда в межах
        for j in range(len(bounds)):
            neighbor[j] = max(bounds[j][0], min(bounds[j][1], neighbor[j]))

        neighbor_value = func(neighbor)

        # Різниця між поточним і сусіднім значеннями
        delta = neighbor_value - value

        # Приймаємо краще рішення або гірше з певною ймовірністю
        if delta < 0 or random.random() < math.exp(-delta / current_temp):
            solution = neighbor
            value = neighbor_value
            
            # Критерій зупинки
            if current_temp < epsilon:
                break

        # Зниження температури
        current_temp *= cooling_rate
        
    return solution, value

# ---
# Функція для побудови 3D графіка функції Сфери та знайдених точок
# ---
def plot_results(func, bounds, hc_sol, hc_val, rls_sol, rls_val, sa_sol, sa_val):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Створення сітки для поверхні функції
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    # Побудова поверхні функції Сфери
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Позначення глобального мінімуму
    ax.scatter(0, 0, 0, color='red', marker='*', s=300, label='Глобальний мінімум (0,0,0)', depthshade=False)

    # Позначення точок, знайдених алгоритмами
    if hc_sol:
        ax.scatter(hc_sol[0], hc_sol[1], hc_val, color='yellow', marker='o', s=100, label='Hill Climbing', depthshade=False)
    if rls_sol:
        ax.scatter(rls_sol[0], rls_sol[1], rls_val, color='green', marker='^', s=100, label='Random Local Search', depthshade=False)
    if sa_sol:
        ax.scatter(sa_sol[0], sa_sol[1], sa_val, color='blue', marker='s', s=100, label='Simulated Annealing', depthshade=False)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f(X1, X2)')
    ax.set_title('Мінімізація функції Сфери різними алгоритмами')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Межі для функції (для двох змінних x1, x2)
    bounds = [(-5, 5), (-5, 5)]

    # Виконання та виведення результатів для Hill Climbing
    print(f"{BLUE}Hill Climbing:{RESET}")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print(f"Розв'язок: {hc_solution}\n{GREEN}Значення:{RESET} {hc_value}")

    print("\n" + f"{YELLOW}={RESET}"*50 + "\n")

    # Виконання та виведення результатів для Random Local Search
    print(f"{BLUE}Random Local Search:{RESET}")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print(f"Розв'язок: {rls_solution}{GREEN}\nЗначення:{RESET} {rls_value}")
    
    print("\n" + f"{YELLOW}={RESET}"*50 + "\n")

    # Виконання та виведення результатів для Simulated Annealing
    print(f"{BLUE}Simulated Annealing:{RESET}")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print(f"Розв'язок: {sa_solution}\n{GREEN}Значення:{RESET} {sa_value}")

    # Побудова графіка
    plot_results(sphere_function, bounds, 
                 hc_solution, hc_value, 
                 rls_solution, rls_value, 
                 sa_solution, sa_value)