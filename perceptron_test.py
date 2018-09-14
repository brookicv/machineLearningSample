
# -*- coding:utf-8 -*-
from pyimagesearch.nn import perceptron
import numpy as np 

def p_or():
    # construct the OR packages
    X = np.array([(0,0),(0,1),(1,0),(1,1)])
    y = np.array([[0],[1],[1],[1]])

    print("[INFO] training perceptron...")
    p = perceptron.Perceptron(X.shape[1],alpha=0.1)
    p.fit(X,y,epochs=20)

    print("[INFO] testing perceptron...")

    for(x,target) in zip(X,y):
        pred = p.predict(x)

        print("[INFO] data={},group-truth={},pred={}".format(x,target[0],pred))


def p_and():
    
    X = np.array([(0,0),(0,1),(1,0),(1,1)])
    y = np.array([[0],[0],[0],[1]])

    print("[INFO] training perceptron...")
    p = perceptron.Perceptron(X.shape[1],alpha=0.1)
    p.fit(X,y,epochs=20)

    print("[INFO] testing perceptron...")

    for(x,target) in zip(X,y):
        pred = p.predict(x)

        print("[INFO] data={},group-truth={},pred={}".format(x,target[0],pred))


def p_xor():
    X = np.array([(0,0),(0,1),(1,0),(1,1)])
    y = np.array([[0],[1],[1],[0]])

    print("[INFO] training perceptron...")
    p = perceptron.Perceptron(X.shape[1],alpha=0.1)
    p.fit(X,y,epochs=20)

    print("[INFO] testing perceptron...")

    for(x,target) in zip(X,y):
        pred = p.predict(x)

        print("[INFO] data={},group-truth={},pred={}".format(x,target[0],pred))


if __name__ == '__main__':

    print('[OR TEST]')
    p_or()

    print('[AND TEST]')
    p_and()

    print('[XOR TEST]')
    p_xor()


    