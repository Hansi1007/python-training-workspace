import numpy as np
import matplotlib.pylab as plt

liste = [1,4,-10,22,7, -1]
print(liste)
x = np.array(liste)
print(x)
print(np.max(x))
print(np.min(x))
print(np.mean(x))
print(np.median(x))


y = np.array([1, 2, 4, 5, 7,8], dtype=np.float32)
print(y)

# scatter
plt.scatter(x, y, color='red')
# plt.plot(x, y, color='b')
plt.legend('f(x)')
plt.title("This is the title")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

