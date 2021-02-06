from math import exp, log


def sigmoid(z):

    """
    The term “sigmoid” means S-shaped, and it is also known as a squashing function, as it maps the whole real range of z into [0,1] in the g(z).
    This simple function has two useful properties that:
    (1) it can be used to model a conditional probability distribution
    (2) its derivative has a simple form.
    It is used to add non-linearity in a machine learning model, since it decides which value to pass as output and what not to pass
    """

    return 1 / (1 + exp(-z))


class Brain:

    def __init__(self, theta=None, layers_size=None, x=None, y=None):

        """
        1. Generate :

        The generate action enables the initilisation of a neural network object.
        The network is composed of nodes grouped in layers and connected by links.
        It is characterised by :

        - layers_count : number of layers in the network
        - layers_size : number of nodes in each layer of the network (excludes bias term *)

        * : during computation a bias term is added at the start of each layer,
        it acts like a constant which helps the model to fit the given data ("ordonnée à l'origine").
        The bias term is NOT included in the layers_size and is Not propagated from one layer to the next.           WHY???????????? :sss

        Within the layers, we can distinguish an input layer, a hidden body of layers and an output layer:

        - layer_0 = x: input/features/variables (1D) - dim = [len(x) x 1]
        - layer_[1:-2] = hiddens : activation nodes (2D) with h_i : hidden layer i (1D) - dim = [layers_size[i]]
        - layer_-1 = y: output/target (1D) - dim = [len(x) x 1]

        The neighbouring layers are connected by steps, each step is made of links connecting all nodes from one layer to the next.
        The parameter matrix theta assembles all network links in a 3D matrix:

        - theta : network parameters matrix (3D) - dim = [dim(theta_step) x layers_count]
        - theta_step : step parameters matrix (2D) - dim = [layers_size[i+1] x layers_size[i]]

        """
        """
        Brain object :

        param :
            a. a theta matrix - type list or ndarray
            b. a layers_size list - type list of int
            c. layer_0 and layer_-1 - type list of values

        return :
        Object of class Brain
        """

        if theta:
            self.layers_count = len(theta) + 1
            self.layers_size = [len(theta[0][0]) - 1] + [len(theta_l) for theta_l in theta]
        elif layers_size:
            self.layers_count = len(layers_size)
            self.layers_size = layers_size
        elif x and y:
            self.layers_count = 3
            self.layers_size = [len(x), 2 * len(x), len(y)]
        else:
            print('Not enough arguments to build a brain :')
            print(theta, layers_size, x, y)
        self.theta = theta
        self.nodes = []

    def forward(self, x):
        """
        2. Think

        The think action uses the neural network to propagate values forwards from one layer to the next.
        It uses forward propagation to fill up the values in the nodes of the successive steps.

        """
        """
        
        The forward propagation is used to compute the values of the nodes in the successive layers.
        The node values are obtained by propagating the values in the input layer (x) layer by layer until reaching the output layer (y).
        This is done by calling the "layer-up" function for each step, and requires having a network parameters matrix to link each step.

        Input : x
        Output : full nodes matrix
        """
        if self.theta is None:
            print("I cant think with no theta man!")
            return None
        self.nodes = [x]
        for step in self.theta:
            self.layer_up()

    def layer_up(self):

        """
        The layer up allows to forward propagate values in one layer (l) to the next (l+1).
        Each node in the upper layer is obtained from :
        - a linear combination of the lower layer and the matching row in the step parameters matrix;
        - a sigmoid of the linear combination, to ensure all node values are comprised between 0 and 1

        Input : -
        Output : updated nodes matrix
        """
        # values = a
        last_completed_layer = len(self.nodes) - 1
        theta_l = self.theta[last_completed_layer]
        last_completed_values = self.nodes[last_completed_layer]
        next_layer_values = []
        for i in range(len(theta_l)):
            # z = theta_1*a_1
            z_value = sum([a * b for a, b in zip(theta_l[i], [1] + list(last_completed_values))])
            # a_2 = g(z)
            next_layer_values.append(sigmoid(z_value))
        self.nodes.append(next_layer_values)


    def init_theta(self):

        """
        test

        """
        #creates theta_init randomly
        from random import random
        #on cree une liste vide > de dimension layer
        self.theta = []
        for l in range(self.layers_count - 1):
            theta_l = []
            for i in range(self.layers_size[l+1]):
                ligne = []
                for j in range(1 + self.layers_size[l]):
                    ligne.append((random() - 0.5))
                theta_l.append(ligne)
            self.theta.append(theta_l)

        # self.theta = [[[(random() - 0.5) / 10 for j in range(1 + self.layers_size[l])] for i in range(self.layers_size[l+1])] for l in range(self.layers - 1)]


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
            y, p = ys[data], self.nodes[-1]
            single_cost = - sum([y[i] * log(p[i]) + (1 - y[i]) * log(1 - min(p[i], 0.9999999999999999)) for i in range(self.layers_size[-1])])
            if reg:
                single_cost += reg / 2 * sum([theta_l_i_j * theta_l_i_j for theta_l_i_j in theta])
                #is this wrong? cfr theory > wrong indent/not for all data
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
            grads = [[[([1] + list(self.nodes[lay]))[j] * errors[lay + 1][i] + grads[lay][i][j] for j in range(len(grads[lay][i]))]
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
        errors = [[self.nodes[-1][i] - y[i] for i in range(len(y))]]
        for layer in range(self.layers_count - 2, 0, -1):
            errors.append([self.nodes[layer][j] * (1 - self.nodes[layer][j]) *
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
        for layer in range(self.layers_count - 1):
            sj, si = 1 + self.layers_size[layer], self.layers_size[layer + 1]
            self.theta.append([[flat_theta[sj * i + j] for j in range(sj)] for i in range(si)])
            flat_theta = flat_theta[si * sj:]

    def error_rate(self, xs, ys):
        er, all = 0, len(xs)
        for i in range(all):
            self.forward(xs[i])
            if [int(val + 0.5) for val in self.nodes[-1]] != ys[i]:
                er += 1
        return er, er/all


def test0():
    xs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    ys = [[0, 1], [0, 1], [0, 1], [1, 1]]

    b = Brain(layers_size=[3, 4, 2])
    b.init_theta()

    for x in xs:
        b.forward(x)
        print(b.nodes, [int(val + 0.5) for val in b.nodes[-1]])

    print("-------------------------------------")
    print(b.learn(xs, ys, 0.000001))
    print("-------------------------------------")

    for x in xs:
        b.forward(x)
        print(b.nodes, [int(val + 0.5) for val in b.nodes[-1]])


def test():
    from datahandler import xs, ys, theta
    b = Brain(theta)
    #print(b.error_rate(xs, ys))
    b.learn(xs[:1], ys[:1])
    # full: 1h20 and no result ; for 10: 7mins nothing (x500 : 1min -> 8h20) ; for 1: 6mins nothing (1 min -> 3.5 days)
    # errors? or need quicker process? can we group cost&grad?   -- explore what takes time
    print(b.error_rate(xs, ys))


test()
