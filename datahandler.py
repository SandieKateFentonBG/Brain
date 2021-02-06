import scipy.io as sio

data = sio.loadmat("ex4data1.mat")
xs, ys = data['X'], []
for y in data['y']:
    newy = [0 for elem in range(10)]
    newy[y[0] - 1] = 1
    ys.append(newy)

theta_dict = sio.loadmat("ex4weights.mat")
theta = [theta_dict['Theta1'], theta_dict['Theta2']]


"""
test = sio.loadmat("ex5data1.mat")
print([k for k in test.keys()])"""
