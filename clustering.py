
import netCDF4
from netCDF4 import Dataset
import numpy as np
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join  
import csv
import datetime
import numpy as np
import time
import matplotlib.dates as mdates
from numpy.linalg import norm
import numpy as np; np.random.seed(0)

import seaborn as sns
#ccsemInt='Physical Properties_CoatedCabojet_Atomizer.csv'
#dfInt = pd.read_csv(ccsemInt, sep=',',header=[0],encoding='unicode_escape')


ccsem='Physical PropertiesCoated_cab_o_jet_atomizer_EDX1_Reduced..csv'


df2 = pd.read_csv(ccsem, sep=',',header=[0],encoding='unicode_escape')

from sklearn import preprocessing

X=[preprocessing.normalize([df2['ECD (Î¼m)'].values[:]])[0],preprocessing.normalize([df2['Sphericity'].values[:]])[0],preprocessing.normalize([df2['Aspect Ratio'].values[:]])[0]]
X=np.transpose(X)
clusterN=6
from sklearn.cluster import KMeans

clusterN=4
kmeans = KMeans(n_clusters=clusterN, random_state=0,n_init='auto' ).fit(X)
df2['label']=kmeans.labels_

avgdf=df2.apply(pd.to_numeric, errors='coerce')
avgdf=avgdf.groupby(['label']).mean()
    
#avgdf.to_csv('Meanof'+str(clusterN)+'CLusters.csv')



fil0=df2[df2['label']==0]
fil1=df2[df2['label']==1]
fil2=df2[df2['label']==2]
fil3=df2[df2['label']==3]
fil4=df2[df2['label']==4]

#fil10=df2[df2['label']==10]
#fil11=df2[df2['label']==11]


#fil6=df2[df2['label']==6]
#fil7=df2[df2['label']==7]
#fil3=df2[df2['label']==3]

figLst=[fil0,fil1,fil2,fil3]
colorLst=['red','blue','green','yellow','pink','violet','purple','orange','brown','brown','tan','gold']
#for i in (range(len(figLst))):
#    fig, (ax2) = plt.subplots(1,1,figsize=(35,14))

    #dfInt.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='black')

    ##fil0.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='red')
    ##fil1.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='blue')
    ##fil2.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='green')
    ##fil3.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='yellow')
    ##fil4.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='pink')
    ##fil5.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='violet')
    ##fil6.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='purple')
    ##fil7.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='orange')
    ##fil8.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='brown')
    ##fil9.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='maroon')

#    figLst[i].plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color=colorLst[i])
    
#    dfInt.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='black',label='Atomizer', alpha=0.3)
#    fig.savefig('ClusterImg/Cluster'+str(i)+'.png')


#fil10.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='tan')
#fil11.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='gold')
fig, (ax2) = plt.subplots(1,1,figsize=(14,14))
for i in (range(len(figLst))):
    

    #dfInt.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='black')

    ##fil0.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='red')
    ##fil1.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='blue')
    ##fil2.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='green')
    ##fil3.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='yellow')
    ##fil4.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='pink')
    ##fil5.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='violet')
    ##fil6.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='purple')
    ##fil7.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='orange')
    ##fil8.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='brown')
    ##fil9.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='maroon')

    figLst[i].plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color=colorLst[i],label=str(i),ylim=(0.5,1.05),xlim=(0,1.5))
    
#dfInt.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='black',label='Interstitials', alpha=0.3)
plt.title('Clustering,'+ccsem[19:-6])
fig.savefig('ClusterImg/completeDistribution.png')

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(fil0['ECD (Î¼m)'], fil0['Sphericity'],fil0['Aspect Ratio'] , color = "red")
ax.scatter3D(fil1['ECD (Î¼m)'], fil1['Sphericity'],fil1['Aspect Ratio'] , color = "blue")
ax.scatter3D(fil2['ECD (Î¼m)'], fil2['Sphericity'],fil2['Aspect Ratio'] , color = "green")
ax.scatter3D(fil3['ECD (Î¼m)'], fil3['Sphericity'],fil3['Aspect Ratio'] , color = "yellow")
ax.scatter3D(fil4['ECD (Î¼m)'], fil4['Sphericity'],fil4['Aspect Ratio'] , color = "pink")
#ax.scatter3D(fil5['ECD (Î¼m)'], fil5['Sphericity'],fil5['Aspect Ratio'] , color = "violet")
#ax.scatter3D(fil6['ECD (Î¼m)'], fil6['Sphericity'],fil6['Aspect Ratio'] , color = "purple")
#ax.scatter3D(fil7['ECD (Î¼m)'], fil7['Sphericity'],fil7['Aspect Ratio'] , color = "orange")
#ax.scatter3D(fil8['ECD (Î¼m)'], fil8['Sphericity'],fil8['Aspect Ratio'] , color = "brown")
#ax.scatter3D(fil9['ECD (Î¼m)'], fil9['Sphericity'],fil9['Aspect Ratio'] , color = "maroon")
#ax.scatter3D(fil10['ECD (Î¼m)'], fil10['Sphericity'],fil10['Aspect Ratio'] , color = "tan")
#ax.scatter3D(fil11['ECD (Î¼m)'], fil11['Sphericity'],fil11['Aspect Ratio'] , color = "gold")



#fil3.plot.scatter(x='ECD (Î¼m)',y='Sphericity',ax=ax2,color='grey')
df2.to_csv('clustering.csv')


data = df2

# Identify the remaining elemental weight percentage columns
remaining_elemental_columns = [col for col in data.columns if col.endswith('(Wt%)')]
# Melt the normalized data for box plotting
melted_normalized_data_percent = data.melt(id_vars='label', value_vars=remaining_elemental_columns, 
                                                              var_name='Element', value_name='Normalized Weight %')

# Define the color list for the plot
color_list = ['red', 'blue', 'green', 'yellow', 'pink', 'violet', 'purple', 'orange', 'brown', 'tan', 'gold']

# Create the normalized box plot (in percentage)
fig=plt.figure(figsize=(15, 8))
sns.boxplot(x='Element', y='Normalized Weight %', hue='label', data=melted_normalized_data_percent, palette=color_list)

# Enhance the plot
plt.title('Elemental Wt,'+ccsem[19:-6])
plt.xticks(rotation=45)
plt.xlabel('Element')
plt.ylabel('Normalized Weight Percentage (%)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
fig.savefig('Elemental plot.png')

plt.show()
