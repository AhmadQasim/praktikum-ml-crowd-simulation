import matplotlib.pyplot as plt
import numpy as np
import functools
from task4.entropy_metric import try_error

errors = [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 8e-1, 1e-0]
real_entropy = []
predicted_entropy = []

for error in errors:
    predicted, real = try_error(error)
    real_entropy.append(real)
    predicted_entropy.append(predicted)

plt.plot(errors, predicted_entropy, label='M estimated')
plt.plot(errors, real_entropy, label='M True')

#coef = np.polyfit(errors, [-10.7342, -6.3853, -1.7484, 2.9268], deg=3)
#linspace = np.linspace(0, 1, num=100)
#result = functools.reduce(lambda x, y: x + y, (c * linspace ** power for c, power in zip(coef, range(len(coef)-1, -1, -1))))
#plt.plot(linspace, result, label='fitted curve of M estimated')

plt.title('Entropy by Model Error')
plt.xlabel('Model Error')
plt.ylabel('Entropy')
plt.legend()
plt.savefig('./report/pictures/task4_entropy.png')