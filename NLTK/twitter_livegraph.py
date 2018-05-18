# -*- coding: utf-8 -*-
"""
Created on Fri May 18 01:06:53 2018

@author: shweta
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('twitter-out.txt','r').read()
    lines = graph_data.split('\n')
    
    xs = []
    ys = []
    x =0
    y =0
    
    for l in lines :
        x+=1
        if "pos" in l:
            y+=1
        elif "neg" in l:
            y-=1
        xs.append(x)
        ys.append(y)
        
    ax1.clear()
    ax1.plot(xs, ys)
    
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()