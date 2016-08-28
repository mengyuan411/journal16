import os 
def list_file_addr(rootDir):
  with open('/home/viki/capstoneProject/scikit-cake/common.txt','r') as f:
    mapper = {} 
    for line in iter(f):
      root = rootDir+line.strip('\n')+'/'
      ap = line.strip('\n')
      record = []
      for lists in os.listdir(root): 
        path = root+lists
        #print(lists)
        #path = os.path.join(root,lists)  
        #if os.path.isdir(path): 
        #  list_file_addr(path)
        #else:
        record.append(path)
      record.sort()
      mapper[ap] = record
  return mapper 
