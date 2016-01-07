# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:04:59 2016

@author: cunnia3
"""
import numpy as np
import math
import matplotlib.pyplot as plt

class Attractor:    
    """
    Class used to represent simple spring-damper system
    """
    def __init__(self,proportional = 10,eqPoint=1):
        self.proportional = proportional
        self.damping = math.sqrt(self.proportional) * 2
        
        # initialize states to 0
        self.xp = 0
        self.xv = 0
        self.xa = 0

        self.root = -math.sqrt(self.proportional)
        self.eqPoint = eqPoint

    def perturb(self,position=0,velocity=0):
        self.xp = position
        self.xv = velocity

    def timeResponse(self,t):
        c1 = self.xp
        c2 = self.xv
        return  math.exp(self.root*t)*(c1 + c2*t)        

    def response(self,xIn,vIn,dt=.1):        
        """
        Function to obtain acceleration response        
        
        args
            dt = increment of time to update system
        """
        return -self.proportional*(xIn-self.eqPoint) - self.damping*vIn
        
    def setEqPoint(self,eqPoint):
        self.eqPoint = eqPoint
    
class Gaussian:
    def __init__(self,center=2,variance=1):
        self.center = center
        self.variance = variance
        
    def response(self,x):
        return math.exp(-self.variance*(x-self.center)**2)
        
        
class ForcingFunction:
    """
    Nonlinear forcing function that drives the system to imitate a behavior
    @params
        nbfs = number of basis functions
    """
    def __init__(self,nbfs = 100):
        self.nbfs = nbfs
        self.basisFunctions = []
        for x in range(0,nbfs):
            self.basisFunctions.append(Gaussian())
            
        self.weights=[1] * self.nbfs
            
        # rate that the forcing function's canonical part decays
        self.decay = -2
    
    def cannonicalSystem(self,time):
        return math.exp(self.decay*time)
            
    def train(self,attractor,example,time):
        """
        Provide the forcing function an example and an attractor to modify the 
        combined forcing function and attractor to match the example
        
        ASSUME THAT THE EXAMPLE IS INDEXED BY TIME AND STARTS AT 0
        """
        # SET CENTERS
        # spread the basis functions evenly (set centers of gaussians)
        # done using studywolf blog's example 
        finalTime = time[-1]
        increment = finalTime/(self.nbfs+2) # dont place basis function at start or end
        
        i=1
        for basisFunction in self.basisFunctions:
            basisFunction.center = self.cannonicalSystem(i*increment)
            i+=1
            
        # SET VARIANCES
        for basisFunction in self.basisFunctions:
            basisFunction.variance = self.nbfs/basisFunction.center
            
            
        # FIND WEIGHTS
        # ASSUME UNIFORM SAMPLING FOR EXAMPLE
        exampleVel = np.diff(example)/(time[1]-time[0])
        exampleAccel = np.diff(exampleVel)/(time[1]-time[0])
        exampleVel=np.append(exampleVel,exampleVel[-1])
        exampleAccel=np.append(exampleAccel,[exampleAccel[-1],exampleAccel[-1]])
        
        # FORM F,S,PSI
        attractorAccel = np.array([])        
        g=example[-1]
        y0 = example[0]
        
        # FIND F
        i=0
        for i in range(len(example)):
            response = attractor.response(example[i],exampleVel[i])
            attractorAccel = np.append(attractorAccel,response) 

        F = exampleAccel - attractorAccel
        Farray = F
        
        # FIND WEIGHTS FOR EACH BASIS FUNCTION by finding PSIs and Ss
        basisFunctionNumber = 0
        for basisFunction in self.basisFunctions:
            P=np.zeros((len(example),len(example)))
            S=np.zeros([len(example),1])
            i=0
            for datum in example:
                response = self.cannonicalSystem(time[i])
                S[i,:] = response*(g-y0)
                P[i,i] = basisFunction.response(self.cannonicalSystem(time[i]))
                i+=1
                
            S = np.mat(S)
            P = np.mat(P)
            F = np.mat(F)
            self.weights[basisFunctionNumber]=np.transpose(S)*P*np.transpose(F)/(np.transpose(S)*P*S)
            basisFunctionNumber+=1
            
        # plot results DEBUGGING
        responseFF = self.responseToTimeArray(time)
        plt.figure()
        plt.title('Trained Forcing Function and Desired Forcing Function')
        plt.plot(responseFF,'g-')
        plt.plot(Farray,'r--')
        plt.show()
        
                
    def response(self,time):
        # if an integer is input
        responseWithoutWeight = 0
        responseWithWeight = 0
        i=0
        for basisFunction in self.basisFunctions: 
            responseWithoutWeight += basisFunction.response(self.cannonicalSystem(time))
            responseWithWeight += self.weights[i]*basisFunction.response(self.cannonicalSystem(time))
            i+=1
            
        # TODO ADD SCALING
        return (responseWithWeight/responseWithoutWeight)*self.cannonicalSystem(time)
    
    def responseToTimeArray(self,timeArray):
        totalResponse = np.array([])
        for time in timeArray:
            totalResponse = np.append(totalResponse,self.response(time))
        return totalResponse
        
class DMP:
    def __init__(self):
        self.dim= 1 # number of input channels
        self.ff = ForcingFunction()
        self.attractor = Attractor()
        
        self.example = np.array([])
        self.exampleTime = np.array([])
        
        self.responseAccel = np.array([0])
        self.responseVel = np.array([1])
        self.responsePos = np.array([0])
        
        self.stepTime = .01 #step time
        self.stepNumber = 0
        
    def _discreteDefIntegral(self, sequence, startIndex, endIndex):
        return np.sum(sequence[startIndex:endIndex+1])
    
    def setExample(self, example, exampleTime):
        self.example = example
        self.exampleTime = exampleTime
        self.attractor.eqPoint=example[-1]
        
    def imitate(self):
        self.ff.train(self.attractor,self.example,self.exampleTime)
                
    def step(self):         
        # UPDATE STATE
        currentPos = self.responsePos[-1]
        currentVel = self.responseVel[-1]
        currentAccel = self.responseAccel[-1]
        
        newVel = currentVel + currentAccel*self.stepTime
        self.responseVel = np.append(self.responseVel,newVel)
        
        newPos = currentPos + currentVel*self.stepTime
        self.responsePos = np.append(self.responsePos,newPos)
        
        # CALCULATE NEW ACCEL  
        newAccel = self.attractor.response(currentPos,currentVel,self.stepTime) + self.ff.response(self.stepNumber * self.stepTime)        
        self.responseAccel = np.append(self.responseAccel,newAccel)        
        
        self.stepNumber += 1
        
    
    def run(self, timeToRun):
        while self.stepNumber*self.stepTime < timeToRun:
            self.step()  
        
    
        