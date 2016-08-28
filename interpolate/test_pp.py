from list_file_addr import list_file_addr
import numpy as np
import pandas as pd
import pp

def interpolate(X_ori, Y_ori, period, table):
  x_vals = numpy.linspace(period[0],period[1],period[1]-period[0]+1).astype(int)
  y_interps = scipy.interpolate.spline(X_ori, Y_ori, x_vals)
  return dict(zip(x_vals,y_interps))

def merge_labels(addr, attris):  #where addr is a list of the file address for a certain AP, attris is map identified by client address
  results = []
  factors = []
  delay_file = []
  for addr_line in addr:
    with open(addr_line.strip('\n'),'r') as f:
      delay_file = [i.strip('\n').split(',') for i in iter(f) if i.strip('\n').split(',')==7]
    f.close()
    print(delay_file)
  delay = pandas.DataFrame(list(zip(delay_file[-1:],delay_file[0:2])),columns = ['client','timestamps'])
  delay = {k:g['timestamps'].tolist() for k,g in delay.groupby('client')}
  delay_attri = {k:(attris[k],delay[k]) for k in set(attris.keys()).intersection(delay.keys())}
  for k,v in delay_attri.items():
    index = 0
    while index<len(v[0]):
      attri = v[0][index]
      counter = 0
      delay_sum = 0
      for line in v[1]:
        content = line.strip('\n').split(',')
        if int(attri[0])<=float(content[0]):
          if int(attri[0])+10>=float(content[1]):
            delay_sum += float(content[1])-float(content[0])
            counter += 1
      if counter:
        results.append(delay_sum/counter)
        factors.append(attri)
    index += 1          
  #labels = np.array(results)
  return (factors,results)

def merge_attri(aps,wifi,delay,output):
  print('running merge attributes')
  attributes = []
  labels = []
  for line in aps:
    key = line.strip('\n')
    print(key)
    au_temp = []
    attri_temp = []
    attri = []
    for elem in wifi[key]:
      file_name = elem.split('/')[8]
      file_attri = file_name.split('.')[1]
      if file_attri == 'channel':
        with open(elem,'r') as g:
          x = [int(i.split(',')[0]) for i in iter(g)]
          if x:
            g.seek(0)
            y = [float(i.split(',')[1]) for i in iter(g)]
            print(len(x))
            print(len(y))
            #period = tuple(int(i) for i in file_name.split('.')[0].split('-'))
            period = (min(x),max(x))
            au_temp = interpolate(x,y,period,[])
          else:
            au_temp = []
        g.close()
      else:
        if au_temp:
          with open(elem,'r') as g:
            attri_temp = [i.strip('\n').split(',') for i in iter(g)]
            attri = [i for i in attri_temp if i[0] in au_temp.keys()]
          g.close()
    attri = pandas.DataFrame(list(zip(attri[1:2],attri[0:1]+attri[2:9])),columns=['client','factors'])
    attri = {k:g['factors'].tolist() for k,g in attri.groupby('client')}
    delay_temp, attri_new = merge_labels(delay[key],attri)
    attributes.append(attri_new)
    labels.append(delay_temp)
    print(attri_new)
    print(delay_temp)
    print('length is '+str(len(attri_new)))
  attributes_np = numpy.array(attributes)
  labels_np = numpy.array(labels)
  with open('/home/viki/capstoneProject/interpolate/attris_out_interpolate'+str(output)+'.txt','w') as f1, open('/home/viki/capstoneProject/interpolate/labels_out_interpolate'+str(output)+'.txt','w') as f2:
    np.save(f1,attributes_np)
    np.save(f2,labels_np)
  f1.close()
  f2.close()

def pp_run():
  wifi = list_file_addr('/home/viki/ANM/data/project2/wifi_use/')
  delay = list_file_addr('/home/viki/ANM/data/project2/delay_use/')
  with open('/home/viki/capstoneProject/scikit-cake/common.txt','r') as f:
    ap_name = [i for i in iter(f)]
    ap_split = [[],[],[],[],[],[],[],[]]
    for i in range(0,len(ap_name)):
      ap_split[i%8].append(ap_name[i].strip('\n'))
  f.close()
  print(ap_split)
  threads = []
  job_server = pp.Server()
  for p in range(0,8):
    threads.append(job_server.submit(merge_attri,(ap_split[p],wifi,delay,p,),(merge_labels,interpolate),('numpy','pandas','scipy.interpolate',)))
    threads[p]()
  job_server.print_stats()


if __name__ == '__main__':
  pp_run()
  

