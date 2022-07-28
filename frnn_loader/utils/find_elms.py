import numpy as np
import sys
import matplotlib.pyplot as plt
import os
def find_elm_events_tar(time,y,threshold=None,scheme=None,maxi =500):
   res=[]
   assert(len(time)==len(y))
   if len(time)==0:
      return []
   tar = np.ones(len(time))*500
   if threshold==None:
      if scheme==None:
         #print(np.mean(y))
         threshold=np.mean(y)*3
#   print(threshold)
   previous_end = -100
   during_elm = False
   current_elm={}
   for i,yi in enumerate(y):
       if yi>threshold:
          if during_elm == False:
             if time[i]-previous_end>5 or len(res)==0:
                current_elm['begin']=time[i]
                current_elm['begin_index']=i
                current_elm['max']=yi
                during_elm=True
                #print('detected ELM at',time[i],'ms')
             else:
               # print('Combining two ELM crashes......')
                current_elm=res.pop()
                during_elm=True
                current_elm['max']=yi
                
          else:
             current_elm['max'] = max(yi,current_elm['max'])
       else:
          if during_elm == True:
             during_elm = False
             current_elm['end']=time[i]
             current_elm['end_index']=i
             res.append(current_elm)
             current_elm={}
             previous_end=time[i]
             #print('ELM ended at',time[i],'ms')
             #print('******************************************')

    
   if during_elm == True:
             during_elm = False
             current_elm['end']=time[i]
             current_elm['end_index']=i
             res.append(current_elm)
             current_elm={}
             previous_end=time[i]
   
   #print(len(res),'ELM events detected~~~!!!')
   previous_end =0
   for e in res:
       index_begin = e['begin_index']
       index_end = e['end_index']
       tar[previous_end:index_begin] = time[index_begin]-time[previous_end:index_begin]
       tar[index_begin:index_end] = 0 #during ELM
       previous_end = index_end

   #print(tar.shape)
   return res,tar

