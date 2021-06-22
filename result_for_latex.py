#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:42:55 2020

@author: dipu
"""

overallMeanClassIou =  ['0.595', '0.471', '0.439']

overallMeanWeightedClassIou =  ['0.729', '0.619', '0.577']

overallMeanAvgPixAcc =  ['0.666', '0.543', '0.508']

overallMeanWeightedPixAcc =  ['0.791', '0.689', '0.646']

a = overallMeanClassIou +overallMeanWeightedClassIou + overallMeanAvgPixAcc + overallMeanWeightedPixAcc
a = str(a)



a = a.replace('[', '')
a = a.replace("'", '')
a = a.replace("]", '')
a = a.replace(",", '')


a = a.split()
a = [float(x)*100 for x in a]
a = [round(x*10)/10 for x in a]

print(a)
a = str(a)
a = a.replace('[', '&')
a = a.replace(',', '&')
a = a.replace(']', '')
print(a)