import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from SVM import SVM

# preparation of training data

center1 = (1, 5)
center2 = (3, 1)
distance = 2
size = 10     # size of the positive and size of the negative samples, total amount of data will be 2*size

x1 = np.random.uniform(center1[0], center1[0]+distance, size=size)  # upper points, label +1
y1 = np.random.uniform(center1[1], center1[1]+distance, size=size)

x2 = np.random.uniform(center2[0], center2[0]+distance, size=size)  # lower points, label -1
y2 = np.random.uniform(center2[1], center2[1]+distance, size=size)

x = np.vstack((np.stack((x1, y1), axis=1), np.stack((x2, y2), axis=1)))

y = np.ones(2*size)
y[size:] = -1     # corresponding labels

# instantiation and fitting of SVM model

model = SVM()
model.fit(x, y, 5000, 0.001, 0.4)
print('SVM model fitted successfully.')

print('Real values | Predicted values for the training set x: ')
for i in range(x.shape[0]):
    print('{0} | {1}'.format(int(y[i]), model.predict(x[i])))

# plot results

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(['red', 'blue']))
t = np.arange(0, 9, 0.2)
h = -(model.weight[0]/model.weight[1]*t+model.bias/model.weight[1])     # equation of a line
plt.plot(t, h)
plt.plot(t, h-1/model.weight[1], '--')
plt.plot(t, h+1/model.weight[1], '--')
plt.title('SVM HYPERPLANE')
plt.show()
