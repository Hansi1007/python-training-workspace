import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # x = np.array(x)
    y = x**2 - 4*x + 5
    return y

def f_ableitung(x):
    return 2*x - 4




if __name__ == "__main__":
    print("Main")

    x = 5
    plt.scatter(x, f(x), c="r")
    for i in range(0, 25):
        steigung_x = f_ableitung(x)
        x = x - 0.05 * steigung_x
        plt.scatter(x, f(x), c="r")
        print(x)

        xs = np.arange(-2, 6, 0.1)
        ys = f(xs)
        plt.plot(xs, ys)

    plt.show()

