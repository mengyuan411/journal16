import scipy.interpolate as itp
import matplotlib.pyplot as plt
#import numpy as np
def interpolate_and_plot(X_ori, Y_ori, period):
  x_vals = np.linspace(period[0],period[1],period[1]-period[0]+1)
  y_interps = itp.spline(X_ori, Y_ori, x_vals)
  plot1=plt.plot(X_ori, Y_ori, 'b*',label='original values')
  plot2=plt.plot(x_vals, y_interps, 'r-x',label='interped values')
  plt.axis([1435152497,1435152653,0,1])
  plt.xlabel('x axis')
  plt.ylabel('y axis')
  plt.legend()
  plt.show()
  plt.title('Interpolate Result')
  plt.savefig('interpolate.png')
  return dict(zip(x_vals,y_interps))

def interpolate(X_ori, Y_ori, period, table):
  x_vals = numpy.linspace(period[0],period[1],period[1]-period[0]+1).astype(int)
  y_interps = itp.spline(X_ori, Y_ori, x_vals)
  #if not table:
  #  table = dict(zip(x_vals,y_interps))
  #else:
  #  table = [i[0]+[i[1]] for i in zip(table,y_interps)]
  return dict(zip(x_vals,y_interps))
