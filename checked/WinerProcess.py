# coding: utf-8

import numpy as np
from scipy.stats import norm

class WinerProcess:
    def __init__(self, n):
        self.n = n
        self.xi = np.zeros((0, self.n + 1))  # случайные величины, индексация: (номер отрезка, номер величины)
        self.segment_count = 0  # число определенных отрезков
        self.segment_left_values = np.zeros(1)  # стартовое значение на отрезке

    def _add_segments(self, new_segments_count):
        """
        Генерация траектории на первых new_segments_count отрезках из еще неопределенных 
        """
        
        # Генерируем новый набор случайных величин
        self.xi = np.append(self.xi, norm.rvs(size=(new_segments_count, self.n + 1)), axis=0)
        
        # Чтобы посчитать стартовые значения на новых отрезках, нужно только взять значение первого линейного слагаемого
        # в точке pi на предыдущем отрезке. К этим значениям нужно применить частичные суммы (cumsum) и к каждому
        # добавить стартовое значение в последнем отрезке из уже существующих
        new_left_values = self.xi[-new_segments_count - 1:-1, 0].cumsum() * np.sqrt(np.pi) + self.segment_left_values[-1]
        self.segment_left_values = np.append(self.segment_left_values, new_left_values)
        
        # Обновление числа определенных отрезков
        self.segment_count = self.xi.shape[0]

    def __getitem__(self, times):
        """
        Подсчет значений процесса в точках times
        """
        
        # С numpy.array работать проще и эффективнее
        times = np.array(times)
        
        # Генерируем процесс на дополнительных отрезках, если это необходимо
        max_segment_index = int(times.max() / np.pi)
        if max_segment_index >= self.segment_count:
            self._add_segments(max_segment_index - self.segment_count + 1)
        
        # Для каждого значения времени определим индекс отрезка (все операции покоординатные)
        segment_indexes = (times / np.pi).astype(int)
        
        # Запишем в ответ линейную часть для каждого значения времени
        W = self.xi[segment_indexes, 0] * np.mod(times, np.pi) / np.sqrt(np.pi)
        
        # Тут создаем двумерную матрицу, которая отвечает синусам. Суммируем ее по оси 1 (по номеру величины)
        W += np.sqrt(2 / np.pi) * (np.sin(times[:, np.newaxis] * np.arange(1, self.n + 1)) * self.xi[segment_indexes, 1:] \
                                   / np.arange(1, self.n + 1)).sum(axis=1)
        
        # Добавляем значение начала отрезка
        W += self.segment_left_values[segment_indexes]
        return W

