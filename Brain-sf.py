from math import exp, log

    """
    
    Other theoretic points :
    
    linear regression
    gradient descent
    logistic regression 
    logistic operations
    multiclass classification
    
    """

class Brain:

    """
    1. Generate :

    INPUT : theta/layers_size/l_0,l_-1
    MAIN : init
    OUTPUT : brain object

    -------------------------------------------------------------------------------------------------------------------

    The generate action enables the initilisation of a neural network object.
    The network is composed of nodes grouped in layers and connected by links.
    It is characterised by :

    - layers_count : number of layers in the network
    - layers_size : number of nodes in each layer of the network (excludes bias term *)

    * : during computation a bias term is added at the start of each layer,
    it acts like a constant which helps the model to fit the given data ("ordonnée à l'origine").
    The bias term is NOT included in the layers_size
                                                                                                               #MOVE

    The bias term is only propagated forwards, and not backpropagated; this can be explained:
    - "materially" : all links connected to a bias term exist downstream from the bias;
    - "mathematically" : when computing gradient descent, bias terms "disappear" - "derivative of constant = 0" ;
    - "common sense" : there is no reason to compute an error on a bias

    Within the layers, we can distinguish an input layer, a hidden body of layers and an output layer:

    - layer_0 = x: input/features/variables (1D) - dim = [len(x) x 1]
    - layer_[1:-2] = hiddens : activation nodes (2D) with h_i : hidden layer i (1D) - dim = [layers_size[i]]
    - layer_-1 = y: output/target (1D) - dim = [len(x) x 1]

    The neighbouring layers are connected by steps, each step is made of links connecting all nodes from one layer to the next.
    The parameter matrix theta assembles all network links in a 3D matrix:

    - theta : network parameters matrix (3D) - dim = [dim(theta_step) x layers_count]
    - theta_step : step parameters matrix (2D) - dim = [layers_size[i+1] x [layers_size[i]+1]
    " +1 : the output nodes will not include the bias nodes, while the inputs will"
    ------------------------------------------------------------------------------------------------------------------

    The generate action uses :
        - init to initiate a neural network object.

    """

    def __init__(self, theta=None, layers_size=None, x=None, y=None):

        """

        Init is used to generate a neural network object.

        --------------------------------------------------------------------------------------------------------------

        Input :
            a. a theta matrix - type list or ndarray
            b. a layers_size list - type list of int
            c. layer_0 and layer_-1 - type list of values

        Return :
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

    """
    2. Think

    INPUT : X, theta
    MAIN : forward propagation
    OUTPUT : Y, nodes

    ------------------------------------------------------------------------------------------------------------------
    
    The think action uses the neural network to propagate values forwards from one layer to the next.
    This returns values for (random)activation nodes and (random)predictions depending on the (random)nature of theta.
    These predictions provide a first cost and error for the model.
    
    ------------------------------------------------------------------------------------------------------------------
    
    It uses :
    
    - layer up to forward propagate values in one layer (l) to the next (l+1).
    - forward propagation to fill up the values in the nodes of the successive steps.
    - pump it : to change the format of theta from a list (2D) to an ndarray (3D)      
    - cost to compute the predicted values in the output layer
     
    """

    def layer_up(self):

        """
        Layer up allows to forward propagate values in one layer (l) to the next (l+1).
        Each node in the upper layer is obtained from :
        - a linear combination of the lower layer and the matching row in the step parameters matrix;
        - a sigmoid of the linear combination, to ensure all node values are comprised between 0 and 1

        # values = a
        # z = theta_1*a_1
        # a_2 = g(z)

        --------------------------------------------------------------------------------------------------------------

        Input : -
        Output : updated nodes matrix, with values for latest step

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

    def forward(self, x):

        """

        The forward propagation is used to compute the values of the nodes in the successive layers.
                                                                                                                                Forward is done for 1 X????
        Node values are obtained by propagating values in input layer(x), layer by layer, until reaching output layer(y).
        Propagation of values requires having a network parameters matrix theta, to link each step.
        Propagation of values is done by calling the "layer-up" function for each step.

        --------------------------------------------------------------------------------------------------------------

        forward function uses:
            - layer up : to propagate values from one step to the next

        --------------------------------------------------------------------------------------------------------------

        Input : x
        Output : full nodes matrix

        """
        if self.theta is None:
            print("I cant think with no theta man!")
            return None
        self.nodes = [x]
        for step in self.theta:
            self.layer_up()

    def cost_function(self, xs, ys, theta, reg):

        """
        A hypothesis function (h_theta(x)), is the model representation used to describe our data.
        It is a mathematical function used to approximate our point cloud.
        It is built from:
            - features (x)
            - parameters (theta)
            - bias terms (theta_0, 1,..)
        Hypothesis functions can consist in polynomial functions
        ex : for linear regression  - h_theta(x) = t0 + t1x + t2x(exp2)
        Hypothesis functions can consist in "neural networks functions"
        ex : h_theta(x) = T * X

        Cost function(J) measures the accuracy of our model, "how well the hypothesis fct maps the training data".
        Cost is a function of the error (e = h_theta(x)-y)
        Error is the difference between values predicted with the hypothesis function, and the "actual/true" values.
        Cost value is computed from the error in various ways, one way is to use the mean squared error MSE function.

        When trying to make a model representation of our data, two problems can occur :

        - underfitting or high bias :
        the form of h_theta(x) maps poorly the data trend, due to a function that is too simple, uses too few features..
        "flat model" : way too imprecise, approximations made are excessive, data points are missed out;

        - overfitting or high variance :
        the form of h_theta(x) fits the available data but doesn't generalize well to predict new data,
        often due to a complicated function that creates a lot of unnecessary curves and angles unrelated to the data
        "mountain model": the model bounces up and down and spirals unnecessarily between every point;
        Depending on the scenario, overfitting can be solved by reducing the number of features or by regularization.

        Regularization is used to tackle the problem of overfitting by reducing the variance of the hypothesis function
        It consists in smoothing the hypothesis function by reducing the influence/magnitude of theta parameters,
        without getting rid of parameters and without changing the form of the model.
        It is obtained by introducing a regularization parameter lambda (l) to inflate the price of parameters.
        To ensure minimal cost, if parameters are expensive, their magnitude must be reduced (*)

        Regularization is applied:                                      ???
        - in the formulation of the cost function,
        - in the formulation of gradient descent
        - not on bias term, since it should not be penalized

        "giving a higher price to high theta values, and thus forcing the magnitudes of theta values to be reduced"

        (*) For a target cost, the higher the value of the lambda, the lower the magnitude of theta:
            unregularized cost J_u = f(sum(t_u))
            regularized cost J_r = f(sum(lambda*t_r))
            Minimize J_r ? if lambda >> then t_r <<
            If lambda is too high, their is a risk of underfitting, if lambda is too low, their is a risk of overfitting
            An adequate value of lambda must be found.

        """

        """
        
        Cost function computes the (random)cost corresponding to the (randomly)defined model representation.
        Cost is computed by averaging the costs of ALL training data                                                        ALL??
        Cost should always be minimal, if the cost is too high, the model representation should be improved:
            - by recalibrating theta
            - by changing the regularization
            
        --------------------------------------------------------------------------------------------------------------
        
        Cost function uses:
            - pump it : to change the format of theta from a list (2D) to an ndarray (3D)
            - forward : to fill up the values in the nodes of the successive steps 

        --------------------------------------------------------------------------------------------------------------

        Input : xs, ys, (random)theta, (reg)
        Output : updated nodes matrix

        """
        self.pump_it(theta)
        total = 0
        for data in range(len(xs)):
            self.forward(xs[data])
            y, p = ys[data], self.nodes[-1]
            single_cost = - sum([y[i] * log(p[i]) + (1 - y[i]) * log(1 - min(p[i], 0.9999999999999999)) for i in range(self.layers_size[-1])])
            total += single_cost
        if reg:
            total += reg / 2 * sum([sum([sum([theta_l_i_j * theta_l_i_j for theta_l_i_j in theta_l_i]) for theta_l_i in theta_l]) for theta_l in theta])

                                                                                                                            #!!check this formula
        return total / len(xs)


    """
        
    3. Learn

    INPUT : X, Y
    MAIN : backpropagation
    OUTPUT : theta

    ------------------------------------------------------------------------------------------------------------------

    The learn action uses the neural network to calibrate theta network parameters and to compute predicted values: 
    It builds from an initial (uncalibrated)theta, from which it forward propagates a first cost.
    It then minimizes the cost function by applying backpropagation, and returns the optimal values of theta.
    It then uses theta values to compute output predictions.

    ------------------------------------------------------------------------------------------------------------------

    It uses :
    
     - init_theta : creates a random network parameters matrix if none are provided, required to start learning process
     - learn
     - cost function
     - gradient_bp
     - backward
     - flat it
     - pump it

    """

    def init_theta(self):

        """
        Initialize theta allows to create a theta network matrix with random values.
        Having a theta matrix is mandatory for starting the learning process.

        ---------------------------------------------------------------------------------------------------------------

        Input : -
        Output : theta network matrix

        """
        #creates theta_init randomly
        from random import random
        self.theta = []
        for l in range(self.layers_count - 1):
            theta_l = []
            for i in range(self.layers_size[l+1]):
                ligne = []
                for j in range(1 + self.layers_size[l]):
                    ligne.append((random() - 0.5))
                theta_l.append(ligne)
            self.theta.append(theta_l)

    def backward(self, y):

        """
        The backward propagation is used to compute the errors associated to the nodes in the successive layers.

        Error values are obtained by propagating values from output layer(y), layer by layer, until reaching the first hidden layer(l).
        Propagation of values requires having activation node values a network parameters matrix theta, to link each step.


        --------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------

        Input : -
        Output : theta network matrix

        """

        errors = [[self.nodes[-1][i] - y[i] for i in range(len(y))]]
        for layer in range(self.layers_count - 2, 0, -1):

            #  g'(z) = a * (1-a)

            errors.append([self.nodes[layer][j] * (1 - self.nodes[layer][j]) *
                           sum([errors[-1][i] * self.theta[layer][i][j + 1] for i in range(self.layers_size[layer + 1])])
                           for j in range(self.layers_size[layer])])
        errors.append([])
        return [errors[index] for index in range(len(errors) - 1, -1, -1)]

    def gradient_bp(self, xs, ys, theta):

        """

        In ANN with gradient-based learning methods and backpropagation,
        each of the neural networks weights receives an update proportional to the partial derivative
        of the error function with respect to the current weight in each iteration of training.

        ---------------------------------------------------------------------------------------------------------------

        uses:

        """

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

    def learn(self, xs, ys, reg=0, reset=False):

        """

        ---------------------------------------------------------------------------------------------------------------
        uses:

        - pump-it
        - init_theta
        - gradient_bp
        - minimize

        ---------------------------------------------------------------------------------------------------------------

        Input : -
        Output : theta network matrix

        """

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



    """
    
    4. Helper functions
    
    ------------------------------------------------------------------------------------------------------------------ 

    Activation functions
    
    In an neural network,the AF of a node defines the output of that node given an input or set of inputs. 
    AF are usually used to remap a large range of input values of z into a constrained domain of g(z), such as [0,a]. 
    By remapping input values to 0, they have the power to activate/deactivate nodes, thus inflate/deflate feature weight on the output.
    Adequate AF allow networks to compute non trivial problems with a small number of nodes, common AF include sigmoid and ReLu.
    The choice of adequate AF can depend on the preferred slope of the remapped values( sigmoid slope == inf; Relu slope == 1),
    or on other advantages/disadvantages expressed by the individual AFs.

    Why do we need values in 0-1/0-inf for NN > they are a succession of classification problems > 1 most likely value per step?
    
    """

    def sigmoid(self, z):           #ADDED self?

    """
        A sigmoid function is an activation function, mapping the whole real range of z into [0,1] in the g(z). 
        It's S-shape allows it to work as a squashing function:
         sig(-inf) = 0
         sig (0) = 0.5
         sig(+inf) = 1
        
        Sigmoid function properties :
        - With values comprised in [0,1] sigmoid can be used to model a conditional probability distribution. 
        - its derivative has a simple form g'(z) = a * (1-a) 
        - it is used to add non-linearity in a machine learning model, 
            since it decides which value to pass as output and what not to pass (power to switch on/off activation nodes)
        - subject to vanishing gradient problem (*)
        
        (*)In ANN with gradient-based learning methods and backpropagation,
        each of the neural networks weights receives an update proportional to the partial derivative
        of the error function with respect to the current weight in each iteration of training.
        The vanishing gradient problem is encountered when the gradient grows vanishingly small,
        effectively preventing the weight from changing its value and even stopping the NN from further training.
        On the inverse, the exploding gradient problem occurs when derivatives are excessively high.  

    """

        return 1 / (1 + exp(-z))

    def relu(self, x):

        """

        A Rectifier is an activation function defined as the positive part of its argument.
        A unit employing the rectifier is called a rectified linear unit ReLu.
        Relu (x) = max(0,x)

        """

        return max(0,x)                     #check!!

    """
    ------------------------------------------------------------------------------------------------------------------
    
    Formatting functions
    
    
    """
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


    """
    ------------------------------------------------------------------------------------------------------------------
    
    Testing functions
    
    
    """

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
