import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from fillipov import fillipov, filippovfunc


def jacobians(
    t,
    y,
    params,
):
    # Распаковка параметров
    # z, w, s, l = params

    # Якобиан в области S1 (H(x) > 0)
    # J1 = np.array(
    #     [[-(2 * z * w + l), 1, 0], [-(2 * z * l * w + w * w), 0, 1], [-l * w * w, 0, 0]]
    # )
    J1 = np.array([[0, 1, 0], [-1, 0, np.cos(y[2])], [0, 0, 0]])

    # Якобиан в области S2 (H(x) < 0)
    J2 = J1

    # Градиент градиента H, вектор, нормальный к поверхности разрыва
    d2H = np.zeros_like(J1)

    return J1, J2, d2H


def vectorfields(t, y, params):
    # Распаковка параметров
    # z, w, s, l = params
    F, omega = params

    # Векторное поле в области 1 - H(x) > 0
    # F1 = np.array(
    #     [
    #         -(2 * z * w + l) * y[0] + y[1] - 1,
    #         -(2 * z * w * l + w * w) * y[0] + y[2] - 2 * s,
    #         -l * w * w * y[0] - 1,
    #     ]
    # )
    F1 = np.array([y[1], -y[0] + np.sin(y[2]) - F, omega])

    # Векторное поле в области 2 - H(x) < 0
    # F2 = np.array(
    #     [
    #         -(2 * z * w + l) * y[0] + y[1] + 1,
    #         -(2 * z * w * l + w * w) * y[0] + y[2] + 2 * s,
    #         -l * w * w * y[0] + 1,
    #     ]
    # )
    F2 = np.array([y[1], -y[0] + np.sin(y[2]) + F, omega])
    # Переключающее многообразие
    H = y[1]

    # Вектор нормали к переключающему многообразию
    dH = np.array([0, 1, 0])

    # Плоскость Пуанкаре
    h = y[2] - 2 * np.pi

    # Направление расположения
    hdir = 1

    return F1, F2, H, dH, h, hdir


def pfunction(t, y, params):
    return np.array([y[0], y[1], 0])


if __name__ == "__main__":
    z = 0.05
    w = 25
    s = -1
    l = 1
    # params = [z, w, s, l]
    F = 0.4
    omega = 1 / 3
    params = [F, omega]

    # Начальные условия
    # y0 = [0.05, -0.01, 0.1]
    y0 = [1, 2, 0]

    # Время интеграции
    T = 2 * np.pi / omega * 10
    t_span = [0, T]

    options = {"atol": 1e-10, "rtol": 1e-10}

    t, y, te, ye, ie, se = fillipov(
        vfields=vectorfields,
        jacobians=jacobians,
        pfunction=pfunction,
        tspan=t_span,
        y0=y0,
        params=params,
        C=1,
        inopts=options,
    )
    print("y = ", y)
    fig = plt.figure(101)
    ax = fig.add_subplot(111, projection="3d")
    ax2 = fig.add_subplot(111)

    ax.plot(y[:, 1], y[:, 2], y[:, 0], "r-", markersize=1)
    ax2.plot(t, y[:, 0], "r-", markersize=1)

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
