import numpy as np
import matplotlib.pyplot as plt

print("random samples")

my_random_array = np.random.randint(low= 10, high=100, size=5)
print(my_random_array)

my_random_array2 = np.random.uniform(low=0, high=10, size=1000)

count, bins, ignored = plt.hist(my_random_array2, 15, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()


my_random_array2 = np.random.binomial(10, 0.5, 1000)

count, bins, ignored = plt.hist(my_random_array2, 15, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()
