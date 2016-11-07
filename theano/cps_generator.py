import numpy as np
from scipy import spatial
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix

def cps_generator(data,label):
  data_merge = zip(data[0],data[1])
  major_x = np.array([x[0] for x in data_merge if list(x[1])==label[0]])
  minor_x = np.array([x[0] for x in data_merge if list(x[1])==label[1]])
  print(major_x.shape[0])
  p = int(float(major_x.shape[0])/float(minor_x.shape[0])) #4 in this case
  print(p)
  minor_new = []
  tree = spatial.KDTree(major_x)
  for item in minor_x:
    distance, M_neighbors = tree.query(x = item, k = major_x[0].shape[0])
    mat = np.array([[]])
    mat = sum([np.mat(tree.data[x]).T*np.mat(tree.data[x]) for x in M_neighbors])
    mat = mat.astype(float)/float(major_x[0].shape[0])
    eigenvalue,eigenmatrix = np.linalg.eig(mat+0.0001*np.identity(major_x[0].shape[0]))
    param = eigenmatrix*np.sqrt(np.diag(eigenvalue))
    minor_new = minor_new + [np.squeeze(np.asarray(np.dot(param,x)+item.T))  for x in np.random.normal(0,1,(p-1,major_x[0].shape[0]))]+[item]
  X = np.array(list(major_x)+minor_new)
  Y = np.array([label[0]]*major_x.shape[0]+[label[1]]*len(minor_new))
  X_sparse = coo_matrix(X)
  X, X_sparse, Y = shuffle(X, X_sparse, Y, random_state=80)
  return (X,Y)

if __name__ == '__main__':
  data = (np.array([[1,1,1],[2,2,2],[1,1,1],[1,1,1]]),np.array([1,2,1,1]))
  label = (1,2)
  m,n = cps_generator(data,label)
  print(m)
  print(n)
  
