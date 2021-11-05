import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

from IPython.display import clear_output
from sklearn.datasets import make_circles

#%%

#CREATE DATASET

#sample size
n = 500
#number of characteristics
p = 2
#Input and  expected output, 0.5 is the distance between the two circles
X, Y = make_circles(n_samples = n, factor = 0.5, noise = 0.05)

Y = Y[:, np.newaxis]

#%%

plt.scatter(X[Y[:, 0] == 0,0], X[Y[:,0] == 0, 1], c = "black")
plt.scatter(X[Y[:, 0] == 1,0], X[Y[:, 0] == 1, 1], c = "salmon")
plt.axis("equal")
plt.show()

#%%

#CLASS NEURAL LAYER

class neural_layer():
    
    def __init__(self, 
                 #num of connections 
                 #from previous to current layer
                 n_conn,
                 #number of neurons in the layer
                 n_neur,
                 #activation function
                 act_f):
        
        self.act_f = act_f
        
        self.b = np.random.rand(1, n_neur)*2 - 1
        self.w = np.random.rand(n_conn, n_neur)*2 - 1
        
# ACTIVATION FUNCTIONS

#sigma function
sigm = (lambda x: 1/(1 + np.e ** (-x)),
        lambda x: x * (1 - x))

#relu function
relu = lambda x: np.maximum(0, x)

#%%
# CREATION OF NEURAL NEWORK

def create_nn(topology, act_f):
    
    nn = []
    
    #index and num of connections
    for l, layer in enumerate(topology[:-1]):
        
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))
    
    return nn

#%%
#  NEURAL TRAINING

# mean square error
l2_cost = (lambda yp, yr: np.mean((yp - yr)**2),
           lambda yp, yr: (yp - yr))

def train(neural_net, X, Y, l2_cost, lr = 0.5, train = True):
    
    out = [(None,X)]
    
    #forward pass
    
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].w + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        
        out.append((z,a))

    
    if train:
        
        #Backwards pass
        
        deltas = []
        
        for l in reversed(range(0, len(neural_net))):
            
            z = out[l + 1][0]
            a = out[l + 1][1]

            if(l ==  len(neural_net) - 1):
                
                #Calculate delta for last layer
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a)) 
                _w = neural_net[l].w #!!!!

            else:

                #Calculate delta for other layers
                deltas.insert(0,  deltas[0] @ _w.T * neural_net[l].act_f[1](a))
            
            _w = neural_net[l].w
            
            #Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis = 0, keepdims=True)*lr

            neural_net[l].w = neural_net[l].w - out[l][1].T @ deltas[0]*lr
            
    return out[-1][1]
#%%
topology = [p, 4, 8, 1]

neural_net = create_nn(topology, sigm)

print(train(neural_net, X, Y, l2_cost))
#%%
topology = [p, 4, 8, 1]
neural_n = create_nn(topology, sigm)

loss = []

for l in range(1000):
    
    pY = train(neural_n, X, Y, l2_cost, lr = 0.1)
    
    if l % 25 == 0:
        loss.append(l2_cost[0](pY, Y))
        
        res = 50
        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
        
        _Y= np.zeros((res,res))
        
        for l0,x0 in enumerate(_x0):
            for l1, x1 in enumerate(_x1):

                _Y[l0,l1] = train(neural_n, np.array([[x0,x1]]), Y, l2_cost, train = False)[0][0]
                
        plt.pcolormesh(_x0, _x1, _Y, cmap = "coolwarm")
        plt.axis("equal")
        
        plt.scatter(X[Y[:, 0] == 0,0], X[Y[:,0] == 0, 1], c = "black")
        plt.scatter(X[Y[:, 0] == 1,0], X[Y[:, 0] == 1, 1], c = "salmon")
        
        clear_output(wait = True)
        plt.show()
        #plt.plot(range(len(loss)), loss)
        #plt.show()
        time.sleep(0.5)
