import numpy as np
def merge_attributes(addr):  #where addr is a list of the file address for a certain AP
  results = []
  length = len(addr)
  index = 0
  while index<length:
    channel = addr[index]
    index +=1
    station = addr[index]
    ap = channel.split('/')[7]
    with open(station.strip('\n'),'r') as f_station, open(channel.strip('\n'),'r') as f_channel:
      for station_line in iter(f_station):
        record_station = station_line.strip('\n').split(',')
        t_station = int(record_station[0])
        for channel_line in iter(f_channel):
          record_channel = channel_line.strip('\n').split(',')
          t_channel = int(record_channel[0])
          if abs(t_station-t_channel)<=9:
            client = record_station[1]
	    timestamp = int(min(record_station[0],record_channel[0]))
            duration = str(10-abs(t_station-t_channel))
            Trx = record_station[2]
            Ttx = record_station[3]
            RR = record_station[4]
            RSSI = record_station[5]
            TPR = record_station[6]
	    RPR = record_station[7]
            AU = record_channel[1]
            UK1 = record_channel[2]
            UK2 = record_channel[3]
            info = []
            info.append(str(timestamp))
            info.append(duration)
            info.append(Trx)
            info.append(Ttx)
            info.append(RR)
            info.append(RSSI)
            info.append(TPR)
            info.append(RPR)
            info.append(ap)
            info.append(client)
            info.append(UK1)
            info.append(UK2)
            results.append(info)
            break
    f_station.close()
    f_channel.close()
    index += 1
  #attributes = np.array(results)
  return results

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
      if content[6] == attri[9]:
        if int(attri[0])<=float(content[0]):
          if int(attri[0])+int(attri[1])>=float(content[1]):
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






 
