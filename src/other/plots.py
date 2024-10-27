from collections.abc import Sequence, Callable
from functools import cached_property
from bisect import bisect_left, bisect_right

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import patches

class Plot:
    def __init__(self, x: Sequence[int | float], y: Sequence[int | float]):
        self.x = x
        self.y = y
        self._point_anomalies = list[tuple[int | float, int | float]]()
        self._struct_anomalies = list[tuple[Sequence[int | float], Sequence[int | float]]]()

    @cached_property
    def density(self) -> float:
        return len(self.x) / (self.x[-1] - self.x[0])

    def add_point_anomaly(self, at: int | float, value: int | float) -> None:
        x = at * self.density
        x -= self.x[0] * self.density
        self.y[int(x)] = value
        self._point_anomalies.append((at, value))

    def add_struct_anomaly(self, x: Sequence[int | float], y: Sequence[int | float]) -> None:
        i = bisect_left(self.x, x[0])
        j = bisect_right(self.x, x[-1])
        self._struct_anomalies.append((self.x[i - 1], self.x[j] if j < len(self.x) else self.x[-1]))
        self.x = np.concatenate([self.x[:i], x, self.x[j:]])
        self.y = np.concatenate([self.y[:i], y, self.y[j:]])


    def plot(self, ax: Axes) -> None:
        ax.plot(self.x, self.y)
        for x, y in self._point_anomalies:
            ax.plot(x, y, "ro")
        for i, j in self._struct_anomalies:
            ax.axvspan(i, j, color='r', alpha=0.25, linewidth=0)


def global_outlier(ax: Axes) -> None:
    """Глобальный выброс"""
    x = np.linspace(0, 10, 200)
    y = np.sin(2 * x)
    plot = Plot(x, y)
    plot.add_point_anomaly(3.5, 1.1)
    plot.plot(ax)
    ax.set_yticks(np.linspace(-1, 1, 5))


def context_outlier(ax: Axes) -> None:
    """Контекстный выброс"""
    x = np.linspace(0, 10, 200)
    y = np.sin(2 * x)
    plot = Plot(x, y)
    plot.add_point_anomaly(3.5, 0.95)
    plot.plot(ax)
    ax.set_yticks(np.linspace(-1, 1, 5))


def shapelet_anomaly(ax: Axes) -> None:
    """Аномалия формы"""
    x = np.linspace(0, 10, 200)
    y = np.sin(4 * x)
    plot = Plot(x, y)
    x = x[90:120]
    y = np.concatenate([np.ones(15), -1 * np.ones(15)])
    plot.add_struct_anomaly(x, y)
    plot.plot(ax)
    ax.set_yticks(np.linspace(-1, 1, 5))

def seasonal_anomaly(ax: Axes) -> None:
    """Сезонная аномалия"""
    x = np.linspace(0, 10, 200)
    y = np.sin(4 * x)
    plot = Plot(x, y)

    x = x[46:62]
    y = np.sin(10 * x)
    plot.add_struct_anomaly(x, y)
    plot.plot(ax)
    ax.set_yticks(np.linspace(-1, 1, 5))


def trend_anomaly(ax: Axes) -> None:
    """Аномалия тренда"""
    x = np.linspace(0, 10, 200)
    y = np.sin(4 * x)
    y[100:] += 1
    plot = Plot(x, y)

    x = x[80:100]
    y = y[80:100]
    diff = 1 / len(y)
    for i in range(len(y)):
        y[i] += diff * (i + 1)
    plot.add_struct_anomaly(x, y)
    plot.plot(ax)
    ax.set_yticks(np.linspace(-1, 2, 7))


def render_all_images() -> None:
    fig, ax = plt.subplots()
    functions: tuple[Callable[[Axes], None]] = (
        global_outlier,
        context_outlier,
        shapelet_anomaly,
        seasonal_anomaly,
        trend_anomaly,
    )
    for f in functions:
        f(ax)
        plt.savefig(f"tmp/{f.__name__}.png", bbox_inches="tight")
        ax.clear()


if __name__ == "__main__":
    render_all_images()
