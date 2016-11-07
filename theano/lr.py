import numpy as np
import theano
import theano.tensor as T
rng = np.random

N = 7817                                   # training sample size
feats = 7                               # number of input variables

# generate a dataset: D = (input_values, target_class)
#D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
with open('/home/viki/capstoneProject/scikit-cake/test2/attris_out.txt','r') as f, open('/home/viki/capstoneProject/scikit-cake/test2/labels_out.txt','r') as g:
  attributes_np = np.load(f)
  labels_np = np.load(g)
  print(attributes_np)
  print(labels_np) 
f.close()
g.close()
labels_new = []
attributes_new = []
for elem in labels_np:
  if elem >= 0.0125:
    labels_new.append(1)
  else:
    labels_new.append(0)
#for elem in attributes_np:
#  timestamps.append(elem[7])
#start = min(timestamps)
#for elem in attributes_np:
#  elem[7] = str((int(elem[7])-int(start))%86400)
labels_np_new = np.array(labels_new)
D = (attributes_np.astype(float),labels_np_new)
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(list(predict(D[0])))
