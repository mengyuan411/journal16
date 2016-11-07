import numpy as np
attributes = []
labels = []
with open('/home/viki/capstoneProject/theano/up.txt','r') as f:
  for line in iter(f):
    content = line.strip('\n').split(',')
    if content[0]:
       attributes.append(content[2:11])
    else:
       delays = [float(i) for i in content[1:]]
       delays_np = np.array(delays)
       ave = np.average(delays_np)
       labels.append(ave)
f.close()
attributes_np = np.array(attributes).astype(float64)
labels_np = np.array(labels).astype(int32)

