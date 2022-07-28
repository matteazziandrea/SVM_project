import numpy as np

# SVM for linearly separable data


class SVM:


    def __init__(self):
        self.x = None
        self.y = None
        self.alpha = None
        self.weight = None
        self.bias = None


    def alpha_gradient(self):   # derivative of the dual lagrangian w.r.t. each of the alpha values, i wanna minimize it
        a_gradient = self.alpha.copy()    # just to have the same size
        n_samples = self.x.shape[0]
        for i in range(n_samples):
            sum = 0
            for j in range(n_samples):
                if j != i:
                    sum += self.y[j]*self.alpha[j]*self.x[i].T.dot(self.x[j])
            sum *= self.y[i]
            sum += self.alpha[i]*np.sum(self.x[i]**2)
            a_gradient[i] = 1-sum     # final i-th derivative
        return a_gradient


    def fit(self, x, y, epoch, learning_rate, grad_eps):
        self.x = x
        self.y = y[:, np.newaxis]
        self.alpha = np.full(self.y.shape, 1, dtype='float')
        for i in range(epoch):
            alpha_gradient = self.alpha_gradient()
            print('Mean of the sum of the absolute values of the derivatives w.r.t. the alpha values: \n{} \n\n'.format(
                np.mean(np.absolute(alpha_gradient))))
            if np.mean(np.absolute(alpha_gradient)) <= grad_eps:  # we have reached the maximum
                self.alpha = np.where(self.alpha > 0.01, self.alpha, 0)
                print('The summation of the alpha values ai times the corresponding yi is: \n{}\n\n'.format(
                    np.sum(self.alpha * self.y)))
                self.set_parameters()
                return None
            alpha_buffer = self.alpha.copy()
            alpha_buffer += learning_rate*self.alpha_gradient()   # gradient ascent
            self.alpha[alpha_buffer >= 0] = alpha_buffer[alpha_buffer >= 0]   # the alpha values must be non-zero values

        self.alpha = np.where(self.alpha > 0.01, self.alpha, 0)   # if an alpha value is very small i set to 0

        print('The summation of the alpha values ai times the corresponding yi is: \n{}\n\n'.format(
            np.sum(self.alpha * self.y)))
        print('Vector of the alpha values: \n{}\n\n'.format(self.alpha))
        self.set_parameters()


    def set_parameters(self):
        self.weight = np.sum(self.alpha*self.y*self.x, axis=0).T
        index = np.argmax(self.alpha)
        self.bias=self.y[index]-self.weight.T.dot(self.x[index])
        print('The vector weight is: \n{0}\n The bias is: \n{1}\n'.format(self.weight, self.bias))


    def predict(self, x):
        if self.weight.dot(x.T) + self.bias>= 0:
            return +1
        else:
            return -1
