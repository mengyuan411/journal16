import numpy as np
from interpolate import interpolate_and_plot,interpolate
with open("/home/viki/ANM/data/project2/wifi_use/6CB0CE0BEB7E/1435075200-1435161600.channel.wlan0.convert","r") as f_ch, open("/home/viki/ANM/data/project2/wifi_use/6CB0CE0BEB7E/1435075200-1435161600.station.wlan0.convert","r") as f_s:
  X_ch = []
  Y_ch = []
  table = np.array([])
  period = (1435152497,1435152653)
  for line in iter(f_ch):
    content = line.strip('\n').split(',')
    X_ch.append(int(content[0]))
    Y_ch.append(float(content[1]))
  table = interpolate(X_ch,Y_ch,period,table)
  table = interpolate(X_ch,Y_ch,period,table)
  print(table)
  #X = []
  #while temp:
  #  try:
  #    temp = ','.join([f_ch.readline().strip('\n'),f_s.readline().strip('\n')])
  #    X.append(temp.split(','))
  #    print(X)
  #  except:
  #    break
	

    
