
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ---------- Task 1: Divide & Conquer min/max ----------

def min_max_divide_and_conquer(arr: List[float]) -> Tuple[float, float]:
    """
    Повертає (мінімум, максимум) з масиву arr, використовуючи підхід 'розділяй і володарюй'.
    Складність O(n). Порожній список не допускається.
    """
    if not arr:
        raise ValueError("Список не може бути порожнім.")

    def helper(lo: int, hi: int) -> Tuple[float, float]:
        # Обробляємо діапазон [lo, hi] включно
        if lo == hi:
            x = arr[lo]
            return (x, x)
        if hi == lo + 1:
            a, b = arr[lo], arr[hi]
            if a < b:
                return (a, b)
            else:
                return (b, a)
        mid = (lo + hi) // 2
        min1, max1 = helper(lo, mid)
        min2, max2 = helper(mid + 1, hi)
        return (min(min1, min2), max(max1, max2))

    return helper(0, len(arr) - 1)


# ---------- Task 2: Greedy optimization for 3D printer ----------

@dataclass
class PrintJob:
    id: str
    volume: float
    priority: int  # 1 (highest), 2, 3 (lowest)
    print_time: int  # minutes (>0)

@dataclass
class PrinterConstraints:
    max_volume: float
    max_items: int

def _to_dataclasses(print_jobs: List[Dict], constraints: Dict) -> tuple[list[PrintJob], PrinterConstraints]:
    jobs: list[PrintJob] = []
    for j in print_jobs:
        # basic validation
        if j["volume"] <= 0 or j["print_time"] <= 0:
            raise ValueError(f"Некоректні дані задачі: {j}")
        if j["priority"] not in (1,2,3):
            raise ValueError(f"Пріоритет повинен бути 1,2,3: {j}")
        jobs.append(PrintJob(**j))
    pc = PrinterConstraints(**constraints)
    if pc.max_volume <= 0 or pc.max_items <= 0:
        raise ValueError("Обмеження принтера повинні бути > 0.")
    return jobs, pc

def optimize_printing(print_jobs: List[Dict], constraints: Dict) -> Dict:
    """
    Оптимізує чергу 3D-друку згідно з пріоритетами та обмеженнями принтера.
    Стратегія (жадібна, стабільна):
      1) Сортуємо завдання за пріоритетом (1 -> 3), зберігаючи вихідний порядок всередині пріоритетів.
      2) Одним проходом формуємо 'партії' (batch) для одночасного друку:
         додаємо наступну модель у поточну партію, якщо не перевищуються max_volume і max_items;
         інакше завершуємо партію та починаємо нову.
      3) Час партії = максимальний print_time серед моделей у ній.
      4) Загальний час = сума часів усіх партій.
      5) Порядок друку — це плоский список ID у послідовності партій, зберігаючи стабільний порядок.
    Цей підхід узгоджується з очікуваними результатами тестів у завданні.
    """
    jobs, pc = _to_dataclasses(print_jobs, constraints)

    # 1) стабільне сортування за пріоритетом
    jobs_sorted = sorted(enumerate(jobs), key=lambda t: (t[1].priority, t[0]))
    jobs_sorted = [j for _, j in jobs_sorted]

    batches: list[list[PrintJob]] = []
    cur_batch: list[PrintJob] = []
    cur_vol = 0.0

    for job in jobs_sorted:
        fits_items = (len(cur_batch) + 1) <= pc.max_items
        fits_volume = (cur_vol + job.volume) <= pc.max_volume

        if cur_batch and not (fits_items and fits_volume):
            # закриваємо поточну партію
            batches.append(cur_batch)
            cur_batch = []
            cur_vol = 0.0

        # почати/продовжити партію
        cur_batch.append(job)
        cur_vol += job.volume

        # якщо партія заповнилася по кількості рівно — закриваємо її негайно
        if len(cur_batch) == pc.max_items:
            batches.append(cur_batch)
            cur_batch = []
            cur_vol = 0.0

    if cur_batch:
        batches.append(cur_batch)

    total_time = 0
    print_order: list[str] = []
    for batch in batches:
        batch_time = max(j.print_time for j in batch)
        total_time += batch_time
        # виводимо ID у стабільному порядку
        print_order.extend(j.id for j in batch)

    return {
        "print_order": print_order,
        "total_time": total_time
    }


# ---------- Tests ----------

def test_task1():
    assert min_max_divide_and_conquer([3]) == (3, 3)
    assert min_max_divide_and_conquer([3, 1]) == (1, 3)
    assert min_max_divide_and_conquer([7, -2, 5, 0, 9, 9, -10]) == (-10, 9)

def test_printing_optimization():
    # Тест 1: Моделі однакового пріоритету
    test1_jobs = [
        {"id": "M1", "volume": 100, "priority": 1, "print_time": 120},
        {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},
        {"id": "M3", "volume": 120, "priority": 1, "print_time": 150}
    ]

    # Тест 2: Моделі різних пріоритетів
    test2_jobs = [
        {"id": "M1", "volume": 100, "priority": 2, "print_time": 120},  # лабораторна
        {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},   # дипломна
        {"id": "M3", "volume": 120, "priority": 3, "print_time": 150}   # особистий проєкт
    ]

    # Тест 3: Перевищення обмежень об'єму
    test3_jobs = [
        {"id": "M1", "volume": 250, "priority": 1, "print_time": 180},
        {"id": "M2", "volume": 200, "priority": 1, "print_time": 150},
        {"id": "M3", "volume": 180, "priority": 2, "print_time": 120}
    ]

    constraints = {
        "max_volume": 300,
        "max_items": 2
    }

    print("Тест 1 (однаковий пріоритет):")
    result1 = optimize_printing(test1_jobs, constraints)
    print(f"Порядок друку: {result1['print_order']}")
    print(f"Загальний час: {result1['total_time']} хвилин")

    print("\nТест 2 (різні пріоритети):")
    result2 = optimize_printing(test2_jobs, constraints)
    print(f"Порядок друку: {result2['print_order']}")
    print(f"Загальний час: {result2['total_time']} хвилин")

    print("\nТест 3 (перевищення обмежень):")
    result3 = optimize_printing(test3_jobs, constraints)
    print(f"Порядок друку: {result3['print_order']}")
    print(f"Загальний час: {result3['total_time']} хвилин")

if __name__ == "__main__":
    # Run tests and also show explicit expected results for comparison
    test_task1()
    test_printing_optimization()
