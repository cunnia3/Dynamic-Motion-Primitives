# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:44:06 2016

@author: cunnia3
"""
from DMPLib import DMP
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, np.pi/2, 0.01)
y = t

testDMP= DMP()
testDMP.setExample(y,t)
testDMP.imitate()
testDMP.run(np.pi/2)

result = testDMP.responsePos

plt.figure()
plt.title('Desired Trajectory and Imitated Trajectory')
plt.plot(result)
plt.plot(y,'g--')
plt.show()
