import numpy as np
a = 10

print('------%d---'%(a))

a = np.arange(8).reshape([2,2,2])
print(a)
print('-------------')
print(a.transpose(1,2,0))