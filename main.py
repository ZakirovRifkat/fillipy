import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from fillipov import fillipov, filippovfunc


def vdp1(t, y):
    return [y[1], (1 - y[0] ** 2) * y[1] - y[0]]


def jacobians(t, y, params, str):
    # Распаковка параметров
    z, w, s, l = params

    # Якобиан в области S1 (H(x) > 0)
    J1 = np.array(
        [[-(2 * z * w + l), 1, 0], [-(2 * z * l * w + w * w), 0, 1], [-l * w * w, 0, 0]]
    )

    # Якобиан в области S2 (H(x) < 0)
    J2 = J1

    # Градиент градиента H, вектор, нормальный к поверхности разрыва
    d2H = np.zeros_like(J1)

    return J1, J2, d2H


def vectorfields(t, y, params, str):
    # Распаковка параметров
    z, w, s, l = params

    # Векторное поле в области 1 - H(x) > 0
    F1 = np.array(
        [
            -(2 * z * w + l) * y[0] + y[1] - 1,
            -(2 * z * w * l + w * w) * y[0] + y[2] - 2 * s,
            -l * w * w * y[0] - 1,
        ]
    )

    # Векторное поле в области 2 - H(x) < 0
    F2 = np.array(
        [
            -(2 * z * w + l) * y[0] + y[1] + 1,
            -(2 * z * w * l + w * w) * y[0] + y[2] + 2 * s,
            -l * w * w * y[0] + 1,
        ]
    )

    # Переключающее многообразие
    H = y[0]

    # Вектор нормали к переключающему многообразию
    dH = np.array([1, 0, 0])

    # Плоскость Пуанкаре
    h = 1

    # Направление расположения
    dir = 1

    return F1, F2, H, dH, h, dir


if __name__ == "__main__":
    z = 0.05
    w = 25
    s = -1
    l = 1
    params = [z, w, s, l]

    # Начальные условия
    y0 = [0.05, -0.01, 0.1]

    # Время интеграции
    T = 10
    t_span = [0, T]
    t0 = t_span[0]
    options = {"atol": 1e-10, "rtol": 1e-10}

    t, y, te, ye, ie, se = fillipov(
        vfields=vectorfields,
        jacobians=jacobians,
        pfunction=None,
        solver="",
        tspan=t_span,
        y0=y0,
        params=params,
        C=1,
        inopts=options,
    )
    print("y = ", y)
    fig = plt.figure(101)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(y[:, 1], y[:, 2], y[:, 0], "r-", markersize=1)
    ax.set_xlabel("y_2")
    ax.set_ylabel("y_3")
    ax.set_zlabel("y_1")

    # Оси координат
    ax.plot([-2, -2], [-4, 4], [0, 0], "k")
    ax.plot([-2, 2], [-4, -4], [0, 0], "k")
    ax.plot([2, 2], [-4, 4], [0, 0], "k")
    ax.plot([-2, 2], [4, 4], [0, 0], "k")

    ax.view_init(elev=0, azim=270)
    ax.set_box_aspect((1, 1, 1))
    plt.show()
