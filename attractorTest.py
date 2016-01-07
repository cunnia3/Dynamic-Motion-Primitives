# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:44:06 2016

@author: cunnia3
"""
from DMPLib import DMP
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 1, 0.01)
y = t*t

testDMP= DMP()
testDMP.setExample(y,t)
testDMP.imitate()
testDMP.run(1)

result = testDMP.responsePos

plt.figure()
plt.title('Desired Trajectory and Imitated Trajectory')
plt.plot(result)
plt.plot(y,'g--')
plt.show()
