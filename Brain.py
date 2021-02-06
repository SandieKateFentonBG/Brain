from math import exp, log


def sigmoid(z):
    return 1 / (1 + exp(-z))


class Brain:
    """
    on construit la machine !!!
    constitue un réseau de neurones = apprendre (find theta) + prédire (forward)
    classe qui permett de predire des valeurs à partir d'une base de données
        # trois options:
        # soit on a d'entree theta (donc il peut en deduire facile layers, layers_size/pas besoin de les donner)
        # soit on a d'entree layers, layers_size,
        # soit on a d'entree X,Y,
    """
    """ les layer size sont sans bias"""

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

        # "none" theta est optionnel
        # i si il n'y en a pas, le but est de le trouver, s'il y en a le but est de l'utiliser pour prédire # list de theta_l
        #  theta_l = list (dim l+1) de sous_liste (dim l)
        #  theta_l = [[float = valeur theta] dim 1 + layer l]dim layer l+1]

    def init_theta(self):
        #creates theta_init randomly
        from random import random
        #on cree une liste vide > de dimension layer
        self.theta = []
        for l in range(self.layers - 1):
            theta_l = []
            for i in range(self.layers_size[l+1]):
                ligne = []
                for j in range(1 + self.layers_size[l]):
                    ligne.append((random() - 0.5))
                theta_l.append(ligne)
            self.theta.append(theta_l)

        # self.theta = [[[(random() - 0.5) / 10 for j in range(1 + self.layers_size[l])] for i in range(self.layers_size[l+1])] for l in range(self.layers - 1)]

    def forward(self, x):
        if self.theta is None:
            print("I cant think with no theta man!")
            return None
        self.values = [x]
        for step in self.theta:
            self.layer_up()

    def layer_up(self):
        # values = a
        last_completed_layer = len(self.values) - 1
        theta_l = self.theta[last_completed_layer]
        last_completed_values = self.values[last_completed_layer]
        next_layer_values = []
        for i in range(len(theta_l)):
            # z = theta_1*a_1
            z_value = sum([a * b for a, b in zip(theta_l[i], [1] + list(last_completed_values))])
            # a_2 = g(z)
            next_layer_values.append(sigmoid(z_value))
        self.values.append(next_layer_values)

    def learn(self, xs, ys, reg=0, reset=False):
        if len(xs) != len(ys):
            return print("Incoherent number of data for X (%d) and Y (%d)" % (len(xs), len(ys)))
        if self.theta is None or reset:
            self.init_theta()
        # ! self.cost et self.gradient_bp sont des fonctions =/= theta_init sont des valeurs !
        from scipy.optimize import minimize
        #calls scipy (who calls cost, gradient_bp)
        opt_content = minimize(lambda theta: self.cost_function(xs, ys, theta, reg), self.flat_it(), options={'disp': True},
                               method='BFGS', jac=lambda theta: self.gradient_bp(xs, ys, theta))
        self.pump_it(opt_content.x)
        return self.theta

    """
    def scipy_optimize(self, x,y,theta_init,cost, gradient_bp):
        # calls cost, gradient_bp
        minimize(cost, theta_init, method='BFGS', jac=gradient_bp, options={'disp': True})
        #return theta_opt
    """

    def cost_function(self, xs, ys, theta, reg):
        self.pump_it(theta)
        total = 0
        for data in range(len(xs)):
            self.forward(xs[data])
            y, p = ys[data], self.values[-1]
            single_cost = - sum([y[i] * log(p[i]) + (1 - y[i]) * log(1 - min(p[i], 0.9999999999999999)) for i in range(self.layers_size[-1])])
            if reg:
                single_cost += reg / 2 * sum([theta_l_i_j * theta_l_i_j for theta_l_i_j in theta])
                #  single_cost += reg / 2 * sum([sum([sum([theta_l_i_j ^ 2] for theta_l_i_j in theta_l_i) for theta_l_i in theta_l]) for theta_l in self.theta])
            total += single_cost
        return total / len(xs)
        #takes forward
        #return cost

    def gradient_bp(self, xs, ys, theta):
        self.pump_it(theta)
        #on copie la dim de theta pour construire notre matrice de grad
        grads = [[[0 for spot in grad_l_i] for grad_l_i in grad_l] for grad_l in self.theta]
        for index in range(len(xs)):
            #on complete le matrice des gradient avec un premier exemple, ensuite on itere a travers tous les exemples et on ajoute toutes les valeurs et a la fin on divise par m)
            self.forward(xs[index])
            errors = self.backward(ys[index])
            grads = [[[([1] + list(self.values[lay]))[j] * errors[lay + 1][i] + grads[lay][i][j] for j in range(len(grads[lay][i]))]
                      for i in range(len(grads[lay]))] for lay in range(len(grads))]
        return self.flat_it([[[val / len(xs) for val in grad_l_i] for grad_l_i in grad_l] for grad_l in grads])
        #takes forward (copy from cost)

        #takes backward
        #takes compute
        #return error

    # def forward(self, theta, x):
        # return a = activation_nodes
        # ? la def au-dessus autorise de faire forward sans theta mais ici c'est plus la cas, ça change rien?

    def backward(self, y):
        errors = [[self.values[-1][i] - y[i] for i in range(len(y))]]
        for layer in range(self.layers - 2, 0, -1):
            errors.append([self.values[layer][j] * (1 - self.values[layer][j]) *
                           sum([errors[-1][i] * self.theta[layer][i][j + 1] for i in range(self.layers_size[layer + 1])])
                           for j in range(self.layers_size[layer])])
        errors.append([])
        return [errors[index] for index in range(len(errors) - 1, -1, -1)]

    def flat_it(self, grad=None):
        if not grad:
            grad = self.theta
        flat = []
        for d3 in grad:
            for d2 in d3:
                for d1 in d2:
                    flat.append(d1)
        return flat

    def pump_it(self, flat_theta):
        self.theta = []
        for layer in range(self.layers - 1):
            sj, si = 1 + self.layers_size[layer], self.layers_size[layer + 1]
            self.theta.append([[flat_theta[sj * i + j] for j in range(sj)] for i in range(si)])
            flat_theta = flat_theta[si * sj:]

    def error_rate(self, xs, ys):
        er, all = 0, len(xs)
        for i in range(all):
            self.forward(xs[i])
            if [int(val + 0.5) for val in self.values[-1]] != ys[i]:
                er += 1
        return er, er/all


def test0():
    xs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    ys = [[0, 1], [0, 1], [0, 1], [1, 1]]

    b = Brain(layers_size=[3, 4, 2])
    b.init_theta()

    for x in xs:
        b.forward(x)
        print(b.values, [int(val + 0.5) for val in b.values[-1]])

    print("-------------------------------------")
    print(b.learn(xs, ys, 0.000001))
    print("-------------------------------------")

    for x in xs:
        b.forward(x)
        print(b.values, [int(val + 0.5) for val in b.values[-1]])


def test():
    from datahandler import xs, ys, theta
    b = Brain(theta)
    #print(b.error_rate(xs, ys))
    b.learn(xs[:1], ys[:1])
    # full: 1h20 and no result ; for 10: 7mins nothing (x500 : 1min -> 8h20) ; for 1: 6mins nothing (1 min -> 3.5 days)
    # errors? or need quicker process? can we group cost&grad?   -- explore what takes time
    print(b.error_rate(xs, ys))


test()
