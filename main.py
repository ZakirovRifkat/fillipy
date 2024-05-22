import numpy as np
import matplotlib.pyplot as plt
from fillipov import fillipov, findstate


def jacobians(
    t,
    y,
    params,
):
    # Распаковка параметров
    # z, w, s, l = params
    A, B, C, delta = params

    # Якобиан в области S1 (H(x) > 0)
    # J1 = np.array(
    #     [[-(2 * z * w + l), 1, 0], [-(2 * z * l * w + w * w), 0, 1], [-l * w * w, 0, 0]]
    # )
    J1 = np.array(
        [
            [0, 1, 0, 0],
            [-A, -B, 1, 0],
            [0, 0, -C, -1],
            [1 / delta, 0, 0, -1 / delta],
        ]
    )

    # Якобиан в области S2 (H(x) < 0)
    J2 = J1

    # Градиент градиента H, вектор, нормальный к поверхности разрыва
    d2H = np.zeros_like(J1)

    return J1, J2, d2H


def vectorfields(t, y, params):
    # Распаковка параметров
    # z, w, s, l = params
    A, B, C, delta = params

    # Векторное поле в области 1 - H(x) > 0
    # F1 = np.array(
    #     [
    #         -(2 * z * w + l) * y[0] + y[1] - 1,
    #         -(2 * z * w * l + w * w) * y[0] + y[2] - 2 * s,
    #         -l * w * w * y[0] - 1,
    #     ]
    # )
    F1 = np.array(
        [
            y[1],
            -B * y[1] - A * y[0] + y[2] - 1 / 2,
            -C * y[2] - y[3],
            (y[0] - y[3]) / delta,
        ]
    )

    # Векторное поле в области 2 - H(x) < 0
    # F2 = np.array(
    #     [
    #         -(2 * z * w + l) * y[0] + y[1] + 1,
    #         -(2 * z * w * l + w * w) * y[0] + y[2] + 2 * s,
    #         -l * w * w * y[0] + 1,
    #     ]
    # )
    F2 = np.array(
        [
            y[1],
            -B * y[1] - A * y[0] + y[2] + 1 / 2,
            -C * y[2] - y[3],
            (y[0] - y[3]) / delta,
        ]
    )
    # Переключающее многообразие
    H = y[1]

    # Вектор нормали к переключающему многообразию
    dH = np.array([0, 1, 0, 0])

    # Плоскость Пуанкаре
    h = 1

    # Направление расположения
    hdir = 1

    return F1, F2, H, dH, h, hdir


if __name__ == "__main__":
    # params
    A = 1.5
    B = 1
    C = 0
    delta = 1.3
    params = [A, B, C, delta]

    # Начальные условия
    y0 = [0, 0, -0.6, 0]

    # Время интеграции
    T = 200
    t_span = [0, T]

    options = {"atol": 1e-10, "rtol": 1e-10}

    t, y, te, ye, ie, se = fillipov(
        vfields=vectorfields,
        jacobians=jacobians,
        pfunction=None,
        tspan=t_span,
        y0=y0,
        params=params,
        C=1,
        inopts=options,
    )

    fig = plt.figure(101)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(y[:, 2], y[:, 0], y[:, 1], "r-", markersize=1)

    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.view_init(elev=0, azim=270)
    ax.set_box_aspect((1, 1, 1))
    plt.show()
