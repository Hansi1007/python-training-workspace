import numpy as np

l1 = [i for i in range(20)]
print(l1)

l2 = l1[0:20:2]
print(l2)

l3 = l1[0:10]
print(l3)


my_array = np.zeros(shape=(2,2), dtype=np.float32)
print("my_array :", my_array)

my_reshape_array = np.reshape(my_array, newshape=(4,1))
print("Reshape (4,1): ", my_reshape_array)

my_reshape_array = np.reshape(my_array, newshape=(4,))
print("Reshape (4,): ", my_reshape_array)

my_random_array = np.random.randint(0,11,20)
print(my_random_array)
