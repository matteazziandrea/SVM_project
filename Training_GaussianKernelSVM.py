import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from GaussianKernelSVM import GaussianKernelSVM
from mpl_toolkits.mplot3d import Axes3D

# preparation of training data

size = 20     # size of the positive and size of the negative samples, total amount of data will be 2*size
r1 = 2        # radius
r2 = 4

theta = np.linspace(0, 2*np.pi, size)

x1, y1 = r1*np.cos(theta), r1*np.sin(theta)     # positive samples, label +1

x2, y2 = r2*np.cos(theta), r2*np.sin(theta)     # negative samples, label -1

x = np.vstack((np.stack((x1, y1), axis=1), np.stack((x2, y2), axis=1)))

y = np.ones(2*size)
y[size:] = -1     # corresponding labels

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(['red', 'blue']))
plt.title('NON LINEARLY SEPARABLE DATA')
plt.show()

# instantiation and fitting of GaussianKernelSVM model

model = GaussianKernelSVM(2.5)

model.fit(x, y, 5000, 0.001, 0.4)
print('GaussianKernelSVM model fitted successfully.')

print('Real values | Predicted values for the training set x: ')
for i in range(x.shape[0]):
    print('{0} | {1}'.format(int(y[i]), model.predict(x[i])))

# plot results

fig = plt.figure()
ax = Axes3D(fig)

xf = np.arange(-6, 6, 0.1)      # variables for plotting the function
yf = np.arange(-6, 6, 0.1)
dim = 300      # number of random combination of x and y to obtain the surface of the function
xf = np.hstack((xf, np.random.choice(xf, dim)))
yf = np.hstack((yf, np.random.choice(yf, dim)))    # add some random points to draw the surface

zf = []
for i in np.stack((xf, yf), axis=1):
    zf.append(model.function(i))
zf = np.array(zf).ravel()

# ax.plot(x, y, z, zdir='z', label='curve in (x, y, z)')   # to use this comment the new xf and yf obtained by hstack

ax.plot_trisurf(xf, yf, zf, color='green')    # comment this to see the cluster inside the Gaussian Kernel
ax.scatter(x[:size, 0], x[:size, 1], c='red')     # they are hidden inside the plot of the function above
ax.scatter(x[size:, 0], x[size:, 1], c='blue')
plt.show()  # comment this and uncomment the part below to add some testing samples

# creation of test set and prediction
#
# size = 10
# r1 = 1
# r2 = 6
#
# theta = np.linspace(0, 2*np.pi, size)
#
# x1, y1 = r1*np.cos(theta), r1*np.sin(theta)
#
# x2, y2 = r2*np.cos(theta), r2*np.sin(theta)
#
# xp = np.stack((x1, y1), axis=1)
# xn = np.stack((x2, y2), axis=1)
#
#
# ax.scatter(xp[:, 0], xp[:, 1], c='black')
# ax.scatter(xn[:, 0], xn[:, 1], c='black')
# plt.show()
#
# print('This points should be classified as 1:')
# for p in xp:
#     print('{}'.format(model.predict(p)))
#
# print('This points should be classified as -1:')
# for n in xn:
#     print('{}'.format(model.predict(n)))
