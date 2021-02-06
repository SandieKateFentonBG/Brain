from math import exp


def sigmoid(z):
    return 1 / (1 + exp(-z))


class Brain:
    def __init__(self, theta=None, layers_size=None, x=None, y=None):
        if theta:
            self.layers = len(theta) + 1
            self.layers_size = [len(theta[0][0]) - 1] + [len(theta_l) for theta_l in theta]
        elif layers_size:
            self.layers = len(layers_size)
            self.layers_size = layers_size
        elif x and y:
            self.layers = 3
            self.layers_size = [len(x), 2 * len(x), len(y)]
        else:
            print('Not enough arguments to build a brain :')
            print(theta, layers_size, x, y)
        self.theta = theta
        self.values = []

    def forward(self, x):
        if self.theta is None:
            print("I cant think with no theta man!")
            return None
        self.values = [x]
        for lol in range(len(self.theta)):
            self.layer_up()

    def layer_up(self):
        last_layer = len(self.values) - 1
        next_values = []
        for i in range(len(self.theta[last_layer])):
            z = [a * b for a, b in zip(self.theta[last_layer][i], [1] + self.values[last_layer])]
            next_values.append(sigmoid(sum(z)))
        self.values.append(next_values)

    def learn(self, xs, ys, reset=False):
        if len(xs) != len(ys):
            return print("Incoherent number of data for X (%d) and Y (%d)" % (len(xs), len(ys)))
        from scipy.optimize import minimize
        if self.theta is None or reset:
            self.init_theta()
        opt_content = minimize(lambda theta: self.cost_function(xs, ys, theta), self.theta, options={'disp': True},
                               method='BFGS', jac=lambda theta: self.gradient_bp(xs, ys, theta))
        self.theta = opt_content.x
        return self.theta

    def init_theta(self):
        from random import random
        self.theta = [
            [[(random() - 0.5) / 10 for j in range(1 + self.layers_size[la])] for i in range(self.layers_size[la + 1])]
            for la in range(self.layers - 1)]

    def cost_function(self, xs, ys, theta):
        self.theta = theta
        total_cost = 0
        for data in range(len(xs)):
            self.forward(xs[data])
            total_cost += sum([(ys[data][i] - self.values[-1][i]) ^ 2 for i in range(self.layers_size[-1])])
        return total_cost / len(xs)