# -*- coding:utf-8 -*-

from pyimagesearch.nn import neuralnetwork
import numpy as np 

X = np.array([(0,0),(0,1),(1,0),(1,1)])
y = np.array([[0],[1],[1],[0]])

nn = neuralnetwork.NeuralNetwork([2,2,1],alpha=0.5)
nn.fit(X,y,epochs=2000)

# now that our network is trained,loop over the XOR data points
for(x,target) in zip(X,y):
    pred = nn.predict(x)[0][0]
    setp = 1 if pred > 0.5 else 0
    print("[INFO] data={},groud-truth={},pred={:.4f},step={}".format(x,target[0],pred,setp))