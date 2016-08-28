from list_file_addr import list_file_addr
from interpolate import interpolate
import numpy as np

def merge_labels(addr, attris):  #where addr is a list of the file address for a certain AP, attris is a list of attributes(the client addr and timestamp matter)
  results = []
  delay_file = []
  for addr_line in addr:
    with open(addr_line.strip('\n'),'r') as f:
      for line in iter(f):
        if len(line.split(','))==7:
          delay_file.append(line) # read all data from the files
      f.close()
  index = 0
  while index<len(attris):
    attri = attris[index]
    counter = 0
    delay_sum = 0
    for line in delay_file:
      content = line.strip('\n').split(',')
      if content[6] == attri[1]:
        if int(attri[0])<=float(content[0]):
          if int(attri[0])+10>=float(content[1]):
            delay_sum += float(content[1])-float(content[0])
            counter += 1
    if counter:
      results.append(delay_sum/counter)
    else:
      attris[index]
      attris.remove(attris[index])
    index += 1          
  #labels = np.array(results)
  return (attris,results)


wifi = list_file_addr('/home/viki/ANM/data/project2/wifi_use/')
delay = list_file_addr('/home/viki/ANM/data/project2/delay_use/')
attributes = []
labels = []
with open('/home/viki/capstoneProject/scikit-cake/common.txt','r') as f:
  for line in iter(f):
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
            for i in attri_temp:
              if int(i[0]) in au_temp.keys():
                attri.append(i+[au_temp[int(i[0])]])
          g.close()
    delay_temp, attri_new = merge_labels(delay[key],attri)
    attributes.append(attri_new)
    labels.append(delay_temp)
    print(attri_new)
    print(delay_temp)
    print('length is '+str(len(attri_new)))
  attributes_np = np.array(attributes)
  labels_np = np.array(labels)
f.close()
with open('/home/viki/capstoneProject/interpolate/attris_out_interpolate.txt','w') as f1, open('/home/viki/capstoneProject/interpolate/labels_out_interpolate.txt','w') as f2:
  np.save(f1,attributes_np)
  np.save(f2,labels_np)
f1.close()
f2.close()



#def tomap():#according to 
  






  
