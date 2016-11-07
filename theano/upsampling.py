from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import numpy as np
def upsample(data,labels,reference):#data is like (X,Y), labels(label of X, label of Y)
  data_merge = zip(data[0],data[1])
  print(data_merge)
  major_x = [x[0] for x in data_merge if list(x[1])==labels[1-reference]]
  major_y = [x[1] for x in data_merge if list(x[1])==labels[1-reference]]
  minor_x = [x[0] for x in data_merge if list(x[1])==labels[reference]]
  minor_y = [x[1] for x in data_merge if list(x[1])==labels[reference]]
  print(len(minor_y))
  N = int(float(len(major_y))/float(len(minor_y)))
  X = major_x
  Y = major_y
  for t in range(0,N):
    X = X+minor_x
    Y = Y+minor_y
  X_sparse = coo_matrix(X)
  X, X_sparse, Y = shuffle(X, X_sparse, Y, random_state=80)
  return (np.array(X),np.array(Y),N)

if __name__ == '__main__':
  x,y = upsample(([[1],[2],[1]],['1','2','1']),(['1'],['2']),1)  
  print(x)
  print(y) 
