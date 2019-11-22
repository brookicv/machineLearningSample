# -*- coding:utf-8 -*-

import numpy as np 

class NeuralNetwork:
    def __init__(self,layers,alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0,len(layers) - 2):
            w = np.random.randn(layers[i] + 1,layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1,layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetWork:{}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self,x):
        return x * (1 - x)

    # backpropagation
    def fit_partial(self,x,y):
        A = [np.atleast_2d(x)]
        # FEEDFORWARD
        # loop over the layers in the network
        for layer in np.arange(0,len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activate and 
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying our
            # nonlinear activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output ,add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first pashe of backpropagation is to compute the
        # difference between our *prediction*(the final output
        #  activation in the activations list) and the true target value
        error = A[-1] - y
        
        # from here,we need to apply the chain rule and build our
        # list of deltas ; the first entry in the delats is simply
        # the error of the output layer times the derivative
        # of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # once your understand the chain rule it becomes super easy
        # to implement with a 'for' loop -- simply loop over the layers in 
        # reverse order(ignoring the last two since we already have taken
        # them into account)
        for layer in np.arange(len(A) - 2,0 ,-1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer,followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        D = D[::-1]

            # WEIGHT UPDATE PHASE
            # loop over the layers
        for layer in np.arange(0,len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas,then multiplying
            # this value by some small lenring rate and adding to our
            # weight matrix -- this is where the actual "learning" takes
            # place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])



    def fit(self,X,y,epochs=1000,displayUpdate=100):
        X = np.c_[X,np.ones((X.shape[0]))]

        for epoch in np.arange(0,epochs):
            for(x,target) in zip(X,y):
                self.fit_partial(x,target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={},loss={:.7f}".format(epoch + 1,loss))

    def predict(self,X,addBias=True):
        # initialize the output prediction as input features
        # -- this value will be(forward) propagated through the
        # network to obtain the final prediction
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p,np.ones((p.shape[0]))]

        for layer in np.arange(0,len(self.W)):
                p = self.sigmoid(np.dot(p,self.W[layer]))

        return p 
    def calculate_loss(self,X,targets):

        targets = np.atleast_2d(targets)
        predictions = self.predict(X,addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss 
    