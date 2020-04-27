# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:09:11 2020

@author: Ghulam Mursaleen
"""
import numpy as np
import matplotlib.pyplot as plt
def uniqueish_color():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
N = 50
x = np.loadtxt( 'hw1xtr.dat' )
y = np.loadtxt( 'hw1ytr.dat' )
"""
print(len(x),len(y))


plt.scatter(x, y,color='red')

plt.show()

plt.plot(x, y)

plt.show()"""
"""
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
"""
"""
plt.figure()"""
plt.subplot(111)
plt.xlabel('hw1xtr (GRAPH 1)',color="blue")
plt.ylabel('hw1ytr' , color="purple")
#plt.plot(x,  'bo', y,  'k')
plt.plot(x,  'k', y,  'k',color=uniqueish_color())
plt.savefig('Graph1.png')
"""
plt.subplot(212)
plt.plot(x, y)
plt.show()"""

x1 = np.loadtxt( 'hw1xte.dat' )
y1 = np.loadtxt( 'hw1yte.dat' )
#print(len(x),len(y))
"""

plt.scatter(x1, y1)

plt.show()

plt.plot(x1, y1)

plt.show()"""
"""
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)"""

plt.figure()
plt.subplot(111)
plt.xlabel('hw1xte (GRAPH 2)',color="blue")
plt.ylabel('hw1yte' , color="purple")
plt.plot(x1,  'k', y1,  'k',color=uniqueish_color())
plt.savefig('Graph2.png')
"""
plt.subplot(212)
plt.plot(x1, y1)
plt.show()"""

x2=np.concatenate((x, x1), axis=None)
y2=np.concatenate((y, y1), axis=None)
plt.figure()

px=plt.subplot(111)
plt.xlabel('hw1xte + hw1xtr (GRAPH3) ',color="blue")
plt.ylabel('hw1yte + hw1ytr' , color="purple")
plt.plot(x2,  'k', y2,  'k',color=uniqueish_color())
plt.savefig('Graph3.png')


