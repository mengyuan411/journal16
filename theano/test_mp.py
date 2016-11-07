import numpy as np
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T
from mp import Layer, MLP, gradient_updates_momentum
from sklearn.cross_validation import train_test_split
from upsampling import upsample
from cps_generator import cps_generator

attributes = []
labels = []
with open('../scikit-cake/test2/attris_out.txt','r') as f, open('../scikit-cake/test2/labels_out.txt','r') as g:
  attributes_np = np.load(f)
  labels_np = np.load(g)
  print(len(list(attributes_np)))
  print(len(list(labels_np))) 
f.close()
g.close()
labels_new = []
attributes_new = []
timestamps = []
for elem in labels_np:
  if elem >= 0.0125:
    labels_new.append([0,1])
  else:
    labels_new.append([1,0])
#for elem in attributes_np:
#  timestamps.append(elem[7])
#start = min(timestamps)
#for elem in attributes_np:
#  elem[7] = str((int(elem[7])-int(start))%86400)
labels_np_new = np.array(labels_new)
#bias = np.ones((attributes_np.shape[0],1))
#attributes_final = np.concatenate((attributes_np,bias),axis=1)
X_train, X_test, y_train, y_test = train_test_split(attributes_np.astype(theano.config.floatX), labels_np_new.astype(theano.config.floatX), test_size=0.1, random_state=88)
#X_train,y_train,N = upsample((X_train,y_train),([0.0,1.0],[1.0,0.0]),0)
#X_train,y_train = cps_generator((X_train,y_train),([1.0,0.0],[0.0,1.0]))
X_train,y_train,N = upsample((X_train,y_train),([0,1],[1,0]),0)
X_train = np.concatenate((X_train,np.ones((X_train.shape[0],1),dtype=theano.config.floatX)),axis=1)
X_test = np.concatenate((X_test,np.ones((X_test.shape[0],1),dtype=theano.config.floatX)),axis=1)
#X_train = X_train.T
#X_test = X_test.T
#y_train = y_train.T
#y_test = y_test.T
# First, set the size of each layer (and the number of layers)
# Input layer size is training data dimensionality (2)
# Output size is just 1-d: class label - 0 or 1
# Finally, let the hidden layers be twice the size of the input.
# If we wanted more layers, we could just add another layer size to this list.
layer_sizes = [X_train.shape[1], 100,70,15, 2]
print(X_train.shape[0])
#print(X_train[0])
# Set initial parameter values
W_init = []
b_init = []
activations = []
for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
    # Getting the correct initialization matters a lot for non-toy problems.
    # However, here we can just use the following initialization with success:
    # Normally distribute initial weights
    W_init.append(np.random.randn(n_input, n_output).astype(theano.config.floatX))
    # Set initial biases to 1
    #b_init.append(np.ones(n_output)*0.)
    # We'll use sigmoid activation for all layers
    # Note that this doesn't make a ton of sense when using squared distance
    # because the sigmoid function is bounded on [0, 1].
    activations.append(T.nnet.sigmoid)
# Create an instance of the MLP class
mlp = MLP(W_init, activations)

# Create Theano variables for the MLP input
mlp_input = T.matrix('mlp_input')
# ... and the desired output
mlp_target = T.matrix('mlp_target')
# Learning rate and momentum hyperparameter values
# Again, for non-toy problems these values can make a big difference
# as to whether the network (quickly) converges on a good local minimum.
learning_rate = 0.01
momentum = 0.9
# Create a function for computing the cost of the network given an input
cost = mlp.squared_error(mlp_input, mlp_target)
# Create a theano function for training the network
train = theano.function([mlp_input, mlp_target], cost,allow_input_downcast=True,
                        updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
# Create a theano function for computing the MLP's output given some input
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))
mlp_predict = theano.function([mlp_input],mlp.predict(mlp_input))
# Keep track of the number of training iterations performed
iteration = 0
# We'll only train the network with 20 iterations.
# A more common technique is to use a hold-out validation set.
# When the validation error starts to increase, the network is overfitting,
# so we stop training the net.  This is called "early stopping", which we won't do here.
max_iteration = 2000
while iteration < max_iteration:
    # Train the network using the entire training set.
    # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
    # where only a subset (or a single point) of the training set is used at each iteration.
    # This can also help the network to avoid local minima.
    current_cost = train(X_train, y_train)
    print(mlp.params[0].get_value())
    print('current cost')
    print(current_cost)
    # Get the current network output for all points in the training set
    current_output = mlp_output(X_test)
    print('current output')
    print(current_output)
    # We can compute the accuracy by thresholding the output
    # and computing the proportion of points whose class match the ground truth class.
    #current_prediction = mlp_predict(X_train)
  
    # Plot network output after this iteration
    #plt.figure(figsize=(8, 8))
    #plt.scatter(X[0, :], X[1, :], c=current_output,
    #            lw=.3, s=3, cmap=plt.cm.cool, vmin=0, vmax=1)
    #plt.axis([-6, 6, -6, 6])
    #plt.title('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost), accuracy))
    #plt.show()
    iteration += 1
accuracy = np.mean((current_output > .5) == y_test)
major_rc = float(np.sum(((current_output > .5) == y_test)&(y_test==[1,0])))/float(np.sum(y_test==[1,0]))
minor_rc =  float(np.sum(((current_output > .5) == y_test)&(y_test==[0,1])))/float(np.sum(y_test==[0,1]))
major_pr = float(np.sum(((current_output > .5) == [1,0])&(y_test == [1,0])))/float(np.sum((current_output > .5) == [1,0]))
minor_pr = float(np.sum(((current_output > .5) == [0,1])&(y_test == [0,1])))/float(np.sum((current_output > .5) == [1,0]))
print('accuracy:'+str(accuracy))
print('major_pr:'+str(major_pr))
print('minor_pr:'+str(minor_pr))
print('major_rc:'+str(major_rc))
print('minor_rc:'+str(minor_rc))
print('major class percentile:'+str(np.mean(labels_np_new==[1,0])))
print('minor class percentile:'+str(np.mean(labels_np_new==[0,1])))
print('training set size:'+str(X_train.shape[0]))
#print('N:'+str(N))





