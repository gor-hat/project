from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import platform
from PIL import Image, ImageDraw

fig = [] 
num = 0
num1 = 0 

url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data = pd.read_csv(url)
 
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]
 
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
 
for angle in range(70,210,2):
 
  fig.append(plt.figure())
  num = num + 1 

while num1 < num:
  ax = fig[num1].gca(projection='3d')
  ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
  num1 = num1 + 1
  
  filename='Volcano_step'+str(num1)+'.png'
  plt.savefig(filename, dpi=96)
#  plt.gca()

ax.view_init(30,angle)
 
url1 = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data1 = pd.read_csv(url1)
 
df1=data1.unstack().reset_index()
df1.columns=["X","Y","Z"]
 
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
 
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
#plt.show()
 
surf1=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig1.colorbar( surf1, shrink=0.5, aspect=5)
plt.show()
 
ax1.view_init(30, 45)
#plt.show()
 
ax1.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
#plt.show()

#desk = "C:\\Users\\CaioWilliandaSilvaSa\\Desktop\\Caio\\B. Sant\\Ademar\\animated_volcano.gif"
#open(desk)