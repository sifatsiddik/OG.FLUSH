# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:46:43 2024

@author: Diego
"""

import pandas as pd
import numpy as np

'''Pipe dimensioning using rationale method'''
# Table to get psi, peak factor coefficient from DWA 110
psit =[
    {'Befestigungsgrad': np.nan, 'Gruppe1': 100.0, 'Gruppe1.1': 130.0, 'Gruppe1.2': 180.0,
     'Gruppe1.3': 225.0, 'Gruppe2': 100.0, 'Gruppe2.1': 130.0, 'Gruppe2.2': 180.0, 
     'Gruppe2.3': 225.0, 'Gruppe3': 100.0, 'Gruppe3.1': 130.0, 'Gruppe3.2': 180.0, 
     'Gruppe3.3': 225.0, 'Gruppe4': 100.0, 'Gruppe4.1': 130.0, 'Gruppe4.2': 180.0, 
     'Gruppe4.3': 225.0}, 
    {'Befestigungsgrad': 0.0, 'Gruppe1': 0.0, 'Gruppe1.1': 0.0,'Gruppe1.2': 0.1, 
      'Gruppe1.3': 0.31, 'Gruppe2': 0.1, 'Gruppe2.1': 0.15, 'Gruppe2.2': 0.3,
      'Gruppe2.3': 0.46, 'Gruppe3': 0.15, 'Gruppe3.1': 0.2, 'Gruppe3.2': 0.45,
      'Gruppe3.3': 0.6, 'Gruppe4': 0.2, 'Gruppe4.1': 0.3, 'Gruppe4.2': 0.55, 'Gruppe4.3': 0.75},
    {'Befestigungsgrad': 10.0, 'Gruppe1': 0.09, 'Gruppe1.1': 0.09, 'Gruppe1.2': 0.19, 
      'Gruppe1.3': 0.38, 'Gruppe2': 0.18, 'Gruppe2.1': 0.23, 'Gruppe2.2': 0.37,
      'Gruppe2.3': 0.51, 'Gruppe3': 0.23, 'Gruppe3.1': 0.28, 'Gruppe3.2': 0.5, 
      'Gruppe3.3': 0.64, 'Gruppe4': 0.28, 'Gruppe4.1': 0.37, 'Gruppe4.2': 0.59, 'Gruppe4.3': 0.77}, 
    {'Befestigungsgrad': 20.0, 'Gruppe1': 0.18, 'Gruppe1.1': 0.18, 'Gruppe1.2': 0.27, 
      'Gruppe1.3': 0.44, 'Gruppe2': 0.27, 'Gruppe2.1': 0.31, 'Gruppe2.2': 0.43,
      'Gruppe2.3': 0.56, 'Gruppe3': 0.31, 'Gruppe3.1': 0.35, 'Gruppe3.2': 0.55,
      'Gruppe3.3': 0.67, 'Gruppe4': 0.35, 'Gruppe4.1': 0.43, 'Gruppe4.2': 0.63, 'Gruppe4.3': 0.8},
    {'Befestigungsgrad': 30.0, 'Gruppe1': 0.28, 'Gruppe1.1': 0.28, 'Gruppe1.2': 0.36, 
      'Gruppe1.3': 0.51, 'Gruppe2': 0.35, 'Gruppe2.1': 0.39, 'Gruppe2.2': 0.5,
      'Gruppe2.3': 0.61, 'Gruppe3': 0.39, 'Gruppe3.1': 0.42, 'Gruppe3.2': 0.6,
      'Gruppe3.3': 0.71, 'Gruppe4': 0.42, 'Gruppe4.1': 0.5, 'Gruppe4.2': 0.68, 'Gruppe4.3': 0.82},
    {'Befestigungsgrad': 40.0, 'Gruppe1': 0.37, 'Gruppe1.1': 0.37, 'Gruppe1.2': 0.44,
      'Gruppe1.3': 0.57, 'Gruppe2': 0.44, 'Gruppe2.1': 0.47, 'Gruppe2.2': 0.56, 'Gruppe2.3': 0.66, 
      'Gruppe3': 0.47, 'Gruppe3.1': 0.5, 'Gruppe3.2': 0.65, 'Gruppe3.3': 0.75, 'Gruppe4': 0.5,
      'Gruppe4.1': 0.56, 'Gruppe4.2': 0.72, 'Gruppe4.3': 0.84},
    {'Befestigungsgrad': 50.0, 'Gruppe1': 0.46, 'Gruppe1.1': 0.46,
      'Gruppe1.2': 0.53, 'Gruppe1.3': 0.64, 'Gruppe2': 0.52, 'Gruppe2.1': 0.55,
      'Gruppe2.2': 0.63, 'Gruppe2.3': 0.72, 'Gruppe3': 0.55, 'Gruppe3.1': 0.58,
      'Gruppe3.2': 0.71, 'Gruppe3.3': 0.79, 'Gruppe4': 0.58, 'Gruppe4.1': 0.63, 
      'Gruppe4.2': 0.76, 'Gruppe4.3': 0.87}, 
    {'Befestigungsgrad': 60.0, 'Gruppe1': 0.55, 'Gruppe1.1': 0.55, 
      'Gruppe1.2': 0.61, 'Gruppe1.3': 0.7, 'Gruppe2': 0.6, 'Gruppe2.1': 0.63,
      'Gruppe2.2': 0.7, 'Gruppe2.3': 0.77, 'Gruppe3': 0.62, 'Gruppe3.1': 0.65,
      'Gruppe3.2': 0.76, 'Gruppe3.3': 0.82, 'Gruppe4': 0.65, 'Gruppe4.1': 0.7,
      'Gruppe4.2': 0.8, 'Gruppe4.3': 0.89},
    {'Befestigungsgrad': 70.0, 'Gruppe1': 0.64, 'Gruppe1.1': 0.64, 'Gruppe1.2': 0.7,
      'Gruppe1.3': 0.77, 'Gruppe2': 0.68, 'Gruppe2.1': 0.71, 'Gruppe2.2': 0.76, 'Gruppe2.3': 0.82,
      'Gruppe3': 0.7, 'Gruppe3.1': 0.72, 'Gruppe3.2': 0.81, 'Gruppe3.3': 0.86, 'Gruppe4': 0.72,
      'Gruppe4.1': 0.76, 'Gruppe4.2': 0.84, 'Gruppe4.3': 0.91}, 
    {'Befestigungsgrad': 80.0, 'Gruppe1': 0.74, 'Gruppe1.1': 0.74, 'Gruppe1.2': 0.78,
      'Gruppe1.3': 0.83, 'Gruppe2': 0.77, 'Gruppe2.1': 0.79, 'Gruppe2.2': 0.83,
      'Gruppe2.3': 0.87, 'Gruppe3': 0.78, 'Gruppe3.1': 0.8, 'Gruppe3.2': 0.86,
      'Gruppe3.3': 0.9, 'Gruppe4': 0.8, 'Gruppe4.1': 0.83, 'Gruppe4.2': 0.87, 'Gruppe4.3': 0.93},
    {'Befestigungsgrad': 90.0, 'Gruppe1': 0.83, 'Gruppe1.1': 0.83, 'Gruppe1.2': 0.87,
      'Gruppe1.3': 0.9, 'Gruppe2': 0.86, 'Gruppe2.1': 0.87, 'Gruppe2.2': 0.89, 
      'Gruppe2.3': 0.92, 'Gruppe3': 0.86, 'Gruppe3.1': 0.88, 'Gruppe3.2': 0.91, 
      'Gruppe3.3': 0.93, 'Gruppe4': 0.88, 'Gruppe4.1': 0.89, 'Gruppe4.2': 0.93, 'Gruppe4.3': 0.96}, 
    {'Befestigungsgrad': 100.0, 'Gruppe1': 0.92, 'Gruppe1.1': 0.92, 'Gruppe1.2': 0.95,
      'Gruppe1.3': 0.96, 'Gruppe2': 0.94, 'Gruppe2.1': 0.95, 'Gruppe2.2': 0.96, 'Gruppe2.3': 0.97, 
      'Gruppe3': 0.94, 'Gruppe3.1': 0.95, 'Gruppe3.2': 0.96, 'Gruppe3.3': 0.97, 'Gruppe4': 0.95, 
      'Gruppe4.1': 0.96, 'Gruppe4.2': 0.97, 'Gruppe4.3': 0.98}]

psit = pd.DataFrame(psit)
psit.set_index('Befestigungsgrad',inplace=True)

# Table to get relationsip between full pipe and partially full pipe in velocity to discharge
qfqtt = [{'qtqf': 0.001, 'vt/vv': 0.1725}, {'qtqf': 0.002, 'vt/vv': 0.2113},
          {'qtqf': 0.003, 'vt/vv': 0.2379}, {'qtqf': 0.004, 'vt/vv': 0.2587}, 
          {'qtqf': 0.005, 'vt/vv': 0.2761}, {'qtqf': 0.006, 'vt/vv': 0.2911}, 
          {'qtqf': 0.007, 'vt/vv': 0.3045}, {'qtqf': 0.008, 'vt/vv': 0.3165},
          {'qtqf': 0.009, 'vt/vv': 0.3275}, {'qtqf': 0.01, 'vt/vv': 0.3377}, 
          {'qtqf': 0.011, 'vt/vv': 0.3472}, {'qtqf': 0.012, 'vt/vv': 0.356}, 
          {'qtqf': 0.013, 'vt/vv': 0.3644}, {'qtqf': 0.014, 'vt/vv': 0.3723},
          {'qtqf': 0.015, 'vt/vv': 0.3798}, {'qtqf': 0.016, 'vt/vv': 0.3869},
          {'qtqf': 0.017, 'vt/vv': 0.3938}, {'qtqf': 0.018, 'vt/vv': 0.4003},
          {'qtqf': 0.019, 'vt/vv': 0.4066}, {'qtqf': 0.02, 'vt/vv': 0.4127}, 
          {'qtqf': 0.021, 'vt/vv': 0.4185}, {'qtqf': 0.022, 'vt/vv': 0.4242},
          {'qtqf': 0.023, 'vt/vv': 0.4296}, {'qtqf': 0.024, 'vt/vv': 0.4349},
          {'qtqf': 0.025, 'vt/vv': 0.44}, {'qtqf': 0.026, 'vt/vv': 0.445},
          {'qtqf': 0.027, 'vt/vv': 0.4499}, {'qtqf': 0.028, 'vt/vv': 0.4546}, 
          {'qtqf': 0.029, 'vt/vv': 0.4592}, {'qtqf': 0.03, 'vt/vv': 0.4637}, 
          {'qtqf': 0.031, 'vt/vv': 0.4681}, {'qtqf': 0.032, 'vt/vv': 0.4724},
          {'qtqf': 0.033, 'vt/vv': 0.4765}, {'qtqf': 0.034, 'vt/vv': 0.4806},
          {'qtqf': 0.035, 'vt/vv': 0.4846}, {'qtqf': 0.036, 'vt/vv': 0.4886},
          {'qtqf': 0.037, 'vt/vv': 0.4924}, {'qtqf': 0.038, 'vt/vv': 0.4962},
          {'qtqf': 0.039, 'vt/vv': 0.4999}, {'qtqf': 0.04, 'vt/vv': 0.5035}, 
          {'qtqf': 0.041, 'vt/vv': 0.5071}, {'qtqf': 0.042, 'vt/vv': 0.5106},
          {'qtqf': 0.043, 'vt/vv': 0.514}, {'qtqf': 0.044, 'vt/vv': 0.5174}, 
          {'qtqf': 0.045, 'vt/vv': 0.5207}, {'qtqf': 0.046, 'vt/vv': 0.524}, 
          {'qtqf': 0.047, 'vt/vv': 0.5272}, {'qtqf': 0.048, 'vt/vv': 0.5304},
          {'qtqf': 0.049, 'vt/vv': 0.5335}, {'qtqf': 0.05, 'vt/vv': 0.5366}, 
          {'qtqf': 0.055, 'vt/vv': 0.5513}, {'qtqf': 0.06, 'vt/vv': 0.5651}, 
          {'qtqf': 0.065, 'vt/vv': 0.578}, {'qtqf': 0.07, 'vt/vv': 0.5903}, 
          {'qtqf': 0.075, 'vt/vv': 0.6019}, {'qtqf': 0.08, 'vt/vv': 0.6129}, 
          {'qtqf': 0.085, 'vt/vv': 0.6234}, {'qtqf': 0.09, 'vt/vv': 0.6335}, 
          {'qtqf': 0.095, 'vt/vv': 0.6432}, {'qtqf': 0.1, 'vt/vv': 0.6525},
          {'qtqf': 0.105, 'vt/vv': 0.6614}, {'qtqf': 0.11, 'vt/vv': 0.67},
          {'qtqf': 0.115, 'vt/vv': 0.6784}, {'qtqf': 0.12, 'vt/vv': 0.6864},
          {'qtqf': 0.125, 'vt/vv': 0.6943}, {'qtqf': 0.13, 'vt/vv': 0.7018},
          {'qtqf': 0.135, 'vt/vv': 0.7092}, {'qtqf': 0.14, 'vt/vv': 0.7163},
          {'qtqf': 0.145, 'vt/vv': 0.7233}, {'qtqf': 0.15, 'vt/vv': 0.7301},
          {'qtqf': 0.155, 'vt/vv': 0.7367}, {'qtqf': 0.16, 'vt/vv': 0.7431},
          {'qtqf': 0.165, 'vt/vv': 0.7494}, {'qtqf': 0.17, 'vt/vv': 0.7555},
          {'qtqf': 0.175, 'vt/vv': 0.7615}, {'qtqf': 0.18, 'vt/vv': 0.7674},
          {'qtqf': 0.185, 'vt/vv': 0.7731}, {'qtqf': 0.19, 'vt/vv': 0.7788},
          {'qtqf': 0.195, 'vt/vv': 0.7843}, {'qtqf': 0.2, 'vt/vv': 0.7896}, 
          {'qtqf': 0.205, 'vt/vv': 0.7949}, {'qtqf': 0.21, 'vt/vv': 0.8001},
          {'qtqf': 0.215, 'vt/vv': 0.8052}, {'qtqf': 0.22, 'vt/vv': 0.8102},
          {'qtqf': 0.225, 'vt/vv': 0.8151}, {'qtqf': 0.23, 'vt/vv': 0.8199},
          {'qtqf': 0.235, 'vt/vv': 0.8246}, {'qtqf': 0.24, 'vt/vv': 0.8293},
          {'qtqf': 0.245, 'vt/vv': 0.8338}, {'qtqf': 0.25, 'vt/vv': 0.8383},
          {'qtqf': 0.26, 'vt/vv': 0.8471}, {'qtqf': 0.27, 'vt/vv': 0.8556}, 
          {'qtqf': 0.28, 'vt/vv': 0.8638}, {'qtqf': 0.29, 'vt/vv': 0.8718},
          {'qtqf': 0.3, 'vt/vv': 0.8795}, {'qtqf': 0.31, 'vt/vv': 0.8871}, 
          {'qtqf': 0.32, 'vt/vv': 0.8945}, {'qtqf': 0.33, 'vt/vv': 0.9016},
          {'qtqf': 0.34, 'vt/vv': 0.9086}, {'qtqf': 0.35, 'vt/vv': 0.9154},
          {'qtqf': 0.36, 'vt/vv': 0.922}, {'qtqf': 0.37, 'vt/vv': 0.9284}, 
          {'qtqf': 0.38, 'vt/vv': 0.9347}, {'qtqf': 0.39, 'vt/vv': 0.9409},
          {'qtqf': 0.4, 'vt/vv': 0.9469}, {'qtqf': 0.41, 'vt/vv': 0.9527}, 
          {'qtqf': 0.42, 'vt/vv': 0.9584}, {'qtqf': 0.43, 'vt/vv': 0.964}, 
          {'qtqf': 0.44, 'vt/vv': 0.9695}, {'qtqf': 0.45, 'vt/vv': 0.9749},
          {'qtqf': 0.46, 'vt/vv': 0.9801}, {'qtqf': 0.47, 'vt/vv': 0.9853},
          {'qtqf': 0.48, 'vt/vv': 0.9903}, {'qtqf': 0.49, 'vt/vv': 0.9952},
          {'qtqf': 0.5, 'vt/vv': 1.0}, {'qtqf': 0.51, 'vt/vv': 1.0047}, 
          {'qtqf': 0.52, 'vt/vv': 1.0093}, {'qtqf': 0.53, 'vt/vv': 1.0139}, 
          {'qtqf': 0.54, 'vt/vv': 1.0183}, {'qtqf': 0.55, 'vt/vv': 1.0225}, 
          {'qtqf': 0.56, 'vt/vv': 1.0269}, {'qtqf': 0.57, 'vt/vv': 1.031}, 
          {'qtqf': 0.58, 'vt/vv': 1.0351}, {'qtqf': 0.59, 'vt/vv': 1.0391},
          {'qtqf': 0.6, 'vt/vv': 1.043}, {'qtqf': 0.61, 'vt/vv': 1.0468}, 
          {'qtqf': 0.62, 'vt/vv': 1.0506}, {'qtqf': 0.63, 'vt/vv': 1.0542},
          {'qtqf': 0.64, 'vt/vv': 1.0578}, {'qtqf': 0.65, 'vt/vv': 1.0613}, 
          {'qtqf': 0.66, 'vt/vv': 1.0648}, {'qtqf': 0.67, 'vt/vv': 1.0681}, 
          {'qtqf': 0.68, 'vt/vv': 1.0714}, {'qtqf': 0.69, 'vt/vv': 1.0746}, 
          {'qtqf': 0.7, 'vt/vv': 1.0777}, {'qtqf': 0.71, 'vt/vv': 1.0608}, 
          {'qtqf': 0.72, 'vt/vv': 1.0838}, {'qtqf': 0.73, 'vt/vv': 1.0867}, 
          {'qtqf': 0.74, 'vt/vv': 1.0895}, {'qtqf': 0.75, 'vt/vv': 1.0923}, 
          {'qtqf': 0.76, 'vt/vv': 1.095}, {'qtqf': 0.77, 'vt/vv': 1.0976}, 
          {'qtqf': 0.78, 'vt/vv': 1.1001}, {'qtqf': 0.79, 'vt/vv': 1.1026},
          {'qtqf': 0.8, 'vt/vv': 1.105}, {'qtqf': 0.81, 'vt/vv': 1.1073}, 
          {'qtqf': 0.82, 'vt/vv': 1.1095}, {'qtqf': 0.83, 'vt/vv': 1.1116}, 
          {'qtqf': 0.84, 'vt/vv': 1.1137}, {'qtqf': 0.85, 'vt/vv': 1.1156}, 
          {'qtqf': 0.855, 'vt/vv': 1.1166}, {'qtqf': 0.86, 'vt/vv': 1.1175},
          {'qtqf': 0.865, 'vt/vv': 1.1184}, {'qtqf': 0.87, 'vt/vv': 1.1193},
          {'qtqf': 0.875, 'vt/vv': 1.1201}, {'qtqf': 0.88, 'vt/vv': 1.1209}, 
          {'qtqf': 0.885, 'vt/vv': 1.1217}, {'qtqf': 0.89, 'vt/vv': 1.1225}, 
          {'qtqf': 0.895, 'vt/vv': 1.1233}, {'qtqf': 0.9, 'vt/vv': 1.124}, 
          {'qtqf': 0.905, 'vt/vv': 1.1247}, {'qtqf': 0.91, 'vt/vv': 1.1253},
          {'qtqf': 0.915, 'vt/vv': 1.126}, {'qtqf': 0.92, 'vt/vv': 1.1266},
          {'qtqf': 0.925, 'vt/vv': 1.1271}, {'qtqf': 0.93, 'vt/vv': 1.1277}, 
          {'qtqf': 0.935, 'vt/vv': 1.1282}, {'qtqf': 0.94, 'vt/vv': 1.1285}, 
          {'qtqf': 0.945, 'vt/vv': 1.129}, {'qtqf': 0.95, 'vt/vv': 1.1294}, 
          {'qtqf': 0.955, 'vt/vv': 1.1297}, {'qtqf': 0.95, 'vt/vv': 1.13}, 
          {'qtqf': 0.955, 'vt/vv': 1.1303}, {'qtqf': 0.97, 'vt/vv': 1.1305},
          {'qtqf': 0.975, 'vt/vv': 1.1306}, {'qtqf': 0.98, 'vt/vv': 1.1307}, 
          {'qtqf': 0.985, 'vt/vv': 1.1307}, {'qtqf': 0.99, 'vt/vv': 1.1397},
          {'qtqf': 0.995, 'vt/vv': 1.1306}, {'qtqf': 1.0, 'vt/vv': 1.1304}]

qfqtt = pd.DataFrame(qfqtt)
qfqtt.set_index('qtqf',inplace=True)
#%%


    
# Function to select columns of psi table according to catchment slope in psif2 function
def slf (sl):
    if sl<1:
        a='Gruppe1'
    elif sl<=4:
        a='Gruppe2'
    elif sl<=10:
        a='Gruppe3'
    else:
        a='Gruppe4'
    return a

#Function to find the index of the closest value on any array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#Function to find psi value according to slope, imperviousness, and rain rate in subcatchment
def psif2(slo,imp,df,r):      #df=psit table, r= rr(rain rate)
    # Select columns in psi table according to slope
    gr=slf(slo)
    df2=df.loc[:, df.columns.str.startswith(gr)]
    # Rename columns with first row (rain rate), select row according to imp
    # Find nearest rain rate index
    df2.columns=df2.iloc[0]
    if (imp>100):
        imp=100
    rser=df2.loc[[imp]]
    rrcol=list(rser)
    xx=find_nearest(rrcol,r)
    
    if r >225:
        rt=225
    elif r <100:
        rt=100
    else:
        rt=r
    # If rr is equal to value in table, use it, otherwise interpolate (up or down)
    if rrcol[xx]==rt:
        psi=rser.iat[0,xx]
        
    elif (rrcol[xx]>rt):
        x0=rrcol[xx-1]
        y0=rser.iat[0,xx-1]
        x1=rrcol[xx]
        y1=rser.iat[0,xx]
        psi=round(y0+(rt-x0)*(y1-y0)/(x1-x0),2)
    else:
        x0=rrcol[xx]
        y0=rser.iat[0,xx]
        x1=rrcol[xx+1]
        y1=rser.iat[0,xx+1]    
        psi=round(y0+(rt-x0)*(y1-y0)/(x1-x0),2)

    return psi  


# Function to calculate cumulated Qmax (cumq) and qdw (sqdw) in first iteration
# qcum=cumQ(max), qm=Qmax, qqdw=qdw, sqdw= sqdw (cumulated qdw), 
# fromnode= value column 'FROM_NODE' in df where function is being applied
# db1= pipedf
# Function returns Panda Series 


def cumq(qcum,qm,qqdw,sqdw,gw,sgw,fromnode,db1): 
    # if only qmax is known, cumQ will be qmax
    if (pd.isna(qcum) & pd.notna(qm)):
        cumq=qm
    # if we have a qcum but no  qm, qcum remains unchanged
    elif (pd.notna(qcum) & pd.isna(qm)):
        cumq=qcum
    # if both are nan, sum all values of cumQ from pipedf, that have been already calculated
    # Values are aggregated from db1 where its column 'To_nodes = fromnode
    # Only if all values are not nan!! 
    # In first iteration Qmax values are summed
    elif (pd.isna(qcum) & pd.isna(qm)):
        cumq=db1.loc[db1['TO_NODE'] == fromnode, 'Qmax'].sum(skipna=False)
    elif (pd.notna(qcum) & pd.notna(qm)):
        cumq=qcum
    
    #Same logic to applied qdw
    if (pd.isna(sqdw) & pd.notna(qqdw)):
        cdw=qqdw
    elif (pd.notna(sqdw) & pd.isna(qqdw)):
        cdw=sqdw
    elif (pd.isna(sqdw) & pd.isna(qqdw)):
        cdw=db1.loc[db1['TO_NODE'] == fromnode, 'qdw'].sum(skipna=False)
    elif (pd.notna(sqdw) & pd.notna(qqdw)):
        cdw=sqdw
    
    #Same logic to applied qgw
    if (pd.isna(sgw) & pd.notna(gw)):
        cgw=gw
    elif (pd.notna(sgw) & pd.isna(gw)):
        cgw=sgw
    elif (pd.isna(sgw) & pd.isna(gw)):
        cgw=db1.loc[db1['TO_NODE'] == fromnode, 'qgw'].sum(skipna=False)
    elif (pd.notna(sgw) & pd.notna(gw)):
        cgw=sgw
    return pd.Series([cumq,cdw,cgw])


# Same function but for all further iterations. intead of summing Qmax, sums cumQ and sqdw
def cumq2(qtot,qt,qqdw,sqdw,gw,sgw,fromnode,db1):
    if (pd.isna(qtot) & pd.notna(qt)):
        cumq=qt
    elif (pd.notna(qtot) & pd.isna(qt)):
        cumq=qtot
    elif (pd.isna(qtot) & pd.isna(qt)):
        cumq=db1.loc[db1['TO_NODE'] == fromnode, 'cumQ'].sum(skipna=False)
    elif (pd.notna(qtot) & pd.notna(qt)):
        cumq=qtot
    
    if (pd.isna(sqdw) & pd.notna(qqdw)):
        cdw=qqdw
    elif (pd.notna(sqdw) & pd.isna(qqdw)):
        cdw=sqdw
    elif (pd.isna(sqdw) & pd.isna(qqdw)):
        cdw=db1.loc[db1['TO_NODE'] == fromnode, 'sqdw'].sum(skipna=False)
    elif (pd.notna(sqdw) & pd.notna(qqdw)):
        cdw=sqdw
   
    if (pd.isna(sgw) & pd.notna(gw)):
        cgw=gw
    elif (pd.notna(sgw) & pd.isna(gw)):
        cgw=sgw
    elif (pd.isna(sgw) & pd.isna(gw)):
        cgw=db1.loc[db1['TO_NODE'] == fromnode, 'sqgw'].sum(skipna=False)
    elif (pd.notna(sgw) & pd.notna(gw)):
        cgw=sgw
           
    return pd.Series([cumq,cdw,cgw])    


# Function to sum pipe lengths, cumulated flow times and largest diameter of pipes
# leading into current pipe. 
# ltot = cum length, l = length, stf=cumulated flow time, cd= cumulated diameter
# fromnode= value column 'FROM_NODE' in df where function is being applied
# db1= pipedf
# Function returns Panda Series 
def cuml(ltot,l,stf,fromnode,db1,cd):
    # If no cum values has been calculated, maximum value of cuml is chosen 
    # from pipes flowing into node (to_node) from db1 (only if all values are not nan)
    # Its own length is then added
    # Max cumulated tf and max diameter are also taken
    if(pd.isna(ltot)):
        cuml=db1.loc[db1['TO_NODE'] == fromnode, 'cuml'].max(skipna=False)
        cuml=cuml+l

        sumtf=db1.loc[db1['TO_NODE'] == fromnode, 'sumtf'].max(skipna=False)
        ccd=db1.loc[db1['TO_NODE'] == fromnode, 'D'].max(skipna=False)
    # If values have been been calculated, they remain unchaged
    else:
        cuml=ltot
        sumtf=stf
        ccd=cd
    return pd.Series([cuml,sumtf,ccd])

def fund (qtot,qmax,sqdw,sl,leng,sumtf,tr,ai,zz,rtrz,qtab,cumd,Tfmin,r15):
    '''Function to calculate diameter, it also updates values that are recalculated in 
    iterations within the function: 'sumtf','rtrz','Qmax','Qtot'
    qtot = Qtot, qmax= Qmax, sqdw= sqdw, sl= pipe slope [%],
    sumtf = cumulated tf, tr = initial tr, ai =Adjusted area,zz = design return period,
    rtrz= rain rate for return period and duration,
    qtab= table with ratios of Qfull and Vfull to calculated ones
    df = table with Qfull values according to slope and Diameters
    Function starts with smallest commercial diameter and starts increasing 
    until all conditions are met
    Use ID column as index of dataframes for pandas'''
    ok=False        #ok for Qtot
    okdw=False      #ok for dry weather flow
                
    # initial index of diameter list starting with largest diameter 
    # of all pipes leading into the current one (cumd)
    dd=find_nearest(dia,cumd*1000) 
    
    sl=sl/100
    # Values for reiterations are set to zero 
    Qmax2=0         
    tf2=0
    rtrz2=0
    Qtot2=0
    
    # While loop until both ok are True
    while (ok==False & okdw==False):

        diam=dia[dd]/1000 # Diameter is taken from list and turned into meters
        
        # Using Prandtl-Colebrook formula vful is calculated, then using Manning Qful
        vful=-2*np.log10(0.0015/(3.71*diam)+(2.51*1.01e-6)/(diam*(np.sqrt(2*9.81*diam*sl))))*np.sqrt(2*9.81*diam*sl)
        Qful=(vful*np.pi*diam**2/4)*1000  #l/s
        
        # Ratio of qtot and qfull is calculated. If it is greater than 0.9,
        # a larger diameter is needed. Under 0.9 it enters the conditional.
        qtqf=round(qtot/Qful,3)
        if (qtqf<0.9):

            # To calculate flow time (tff [min]), we find nearest value in table for qtqf
            # and extract vt/vv to calculate vtot and then tff
            b=find_nearest(qtab.index.values, qtqf)
            vtvf=qtab.iloc[b]['vt/vv']
            vtot=vtvf*vful
            tff=leng/(60*vtot)
            
            # If no sumtf was calculated before it is equal to tff, otherwise it is added
            # stf is inside function, sumtf comes from row where function is being applied
            if (np.isnan(sumtf)):
                stf=tff
            else:
                stf=tff+sumtf
            
            # If both  stf and tr are under the minimum time flow, no further calc is needed
            # and ok = True
            # else, delta tf (dtf) is calculated and must be under 5%
            if (stf<Tfmin and tr<Tfmin):
                ok=True
            else:
                dtf=abs((stf-tr)*100/tr)
                if (dtf<5):
                    ok=True
                else:
                    # If dtf is over 5% tf can be used to recalculate discharge
                    # (rtrz2, Qmax2, Qtot2) and use these values to check if dtf meets 
                    # the condition
                    
                    tf2=max(stf,Tfmin)
                    
                    rtrz2=r15*(38/(9+tf2))*(zz**0.25-0.369)            
                    Qmax2=rtrz2*ai
                    # Since Qtot included everythin, but only qmax changed,
                    # qtot can be recalculated by substracting previous qmax and adding new one
                    Qtot2=qtot-qmax+Qmax2
    
                    qtqf=round(Qtot2/Qful,3)
                    b=find_nearest(qtab.index.values, qtqf)
                    vtvf=qtab.iloc[b]['vt/vv']
                    vtot=vtvf*vful
                    
                    tff=leng/(60*vtot)
                    tff=tff+sumtf
                    if (stf<Tfmin and tr<Tfmin):
                        ok=True
                    else:
                        
                        dtf=abs((tff-stf)*100/tff)
                        if (dtf<5):
                            ok=True 
                                    
            # to Check if dwf won't turn into sediment vdw should be higher than 0.5
            # we check using same procedure
            
            if (sqdw==0):
                okdw=True
            else:
                
                qdwqf=round(sqdw/Qful,3)
                
                    
                b=find_nearest(qtab.index.values, qdwqf)
                vdwvf=qtab.iloc[b]['vt/vv']
          
                vdw=vdwvf*vful
            
                if (vdw>0.5 or vdw==0):
                    okdw=True
        # if ant of ok are False, both are set to false and we repeat with larger diameter
        if (ok==False):
            # aqui continua la iteracion y se verifica si se tiene que cambiar diametro
            dd=dd+1
            ok=False
            okdw=False

    # If no reiterations where needed for dtf, we turn the values from zero to the  original values
    if (rtrz2==0):
        rtrz2=rtrz
    if (Qmax2==0):
        Qmax2=qmax
    if (Qtot2==0):
        Qtot2=qtot
        
    
    # return Panda Series of diameter and updated values 
    return pd.Series([diam, stf, rtrz2, Qmax2, Qtot2,okdw])

#Commercial Diameters in mm
dia=(200,250,300,350,400,450,500,600,700,800,900,1000,
      1100,1200,1300,1400,1500,1600,1800,2000,2200,2400,
      2600,2800,3000,3200,3400,3600,3800,4000)
 
# Initial values, constants 


Vest= 2.5

def pipe_dim_prep(pipedf,subdf,psit,r15,qs,fgw,Tfmin,z,it):
    ''' Function to prepare subcathcment contribution to sewer system. join data where needed and
    process the first (head) sections'''
    pipedf.columns = pipedf.columns.astype(str)
    subdf.columns = subdf.columns.astype(str)
    
    pipedf.set_index('LINK_ID',inplace=True)
    subdf.set_index(['NODE_OUTLET'],inplace=True)    
    # Create column for cumQ and cumd asnan
    pipedf['cumQ']=np.nan
    
    #TEMPORARY SET SLOPES TO 0.3% MINIMUM SLOPE
    pipedf['slope']=pipedf.apply(lambda x:max(x['slope'],0.3),axis=1)
    
    # Calculate psi value for each subctachment
    subdf['psi']=subdf.apply(lambda x:psif2(x['slope'],x['%imp'],psit,r15),axis=1)
    
    # Population inh/ha vs total population
    subdf['qdw']=subdf['population']*qs
    subdf['qgw']=subdf['qdw']*fgw
    subdf['Ais']=subdf['AREA']*subdf['psi']
    
    # First iteration
    # From pipedf, rows are selected where no To-node is in From-node
    head=pipedf[-pipedf['FROM_NODE'].isin(pipedf['TO_NODE'])]
    head = head.copy()
    head.loc[:, 'it']=it
    
    # Turn from-node into index and add qdw from corresponding subcatchment, calculate reduced area
    head=head.reset_index().set_index('FROM_NODE')
    head.index = head.index.astype(str)
    subdf.index = subdf.index.astype(str)
    
    aggregated_subdf = subdf[['qdw', 'qgw', 'Ais']].groupby(subdf.index).sum()
    
    # Now join the aggregated DataFrame with head
    head = head.join(aggregated_subdf)
    
    # Fill nan as zero for pipes with no connected area or qdw
    head['Ais']=head['Ais'].fillna(0)
    head['qgw']=head['qgw'].fillna(0)
    head['qdw']=head['qdw'].fillna(0)
    # For first iteration, sqdw and cuml are equal to own values
    head['sqdw']=head.qdw
    head['sqgw']=head.qgw
    head['cuml']=head['length']
    head['cumd']=0
    
    # add tf min, z, calculate tr. sumtf is nan for first iteration
    head['tfmin']=Tfmin
    head['z']=z 
    head['tf']=head.cuml/(Vest*60)
    head['tr']=head[['tfmin','tf']].max(axis=1)
    head['sumtf']=np.nan
    
    # Calculate rainrate according to return period and tr. Then Qmax and Qtot
    head['rtrz']=r15*(38/(9+head.tr))*(head.z**0.25-0.369)
    head['Qmax']=head.rtrz*head.Ais
    head['Qtot']=head.Qmax+head.qdw+head.qgw
    head[['D','sumtf','rtrz','Qmax','Qtot','okqdw']]=head.apply(lambda x:fund(x['Qtot'],x['Qmax'],x['sqdw'],x['slope'],x['cuml'],x['sumtf'],x['tf'],x['Ais'],x['z'],x['rtrz'],qfqtt,x['cumd'],Tfmin,r15),axis=1)
    head=head.reset_index().set_index('LINK_ID')
    
    # Copy values from head to pipedf
    pipedf=pipedf.join(head[['Qtot','it','sqdw','cuml','qdw','qgw','sqgw','Qmax','sumtf','D','okqdw','cumd']])
    
    # Use cumq and cuml functions in pipedf to calculate accumulated Q, qdw, etc
    pipedf[['cumQ','sqdw','sqgw']]=pipedf.apply(lambda x:cumq(x['cumQ'],x['Qmax'],x['qdw'],x['sqdw'],x['qgw'],x['sqgw'],x['FROM_NODE'],pipedf),axis=1)
    pipedf[['cuml','sumtf','cumd']]=pipedf.apply(lambda x:cuml(x['cuml'],x['length'],x['sumtf'],x['FROM_NODE'],pipedf,x['cumd']),axis=1)
    
    return head, pipedf, subdf

def pipe_dimension(pipedf,lastit,subdf,Tfmin,z,r15,it):
    '''Function to dimension the rest of the pipes until all are designed'''
    aggregated_subdf = subdf[['qdw', 'qgw', 'Ais']].groupby(subdf.index).sum()
    aggregated_subdf.index.name = 'FROM_NODE'
    aggregated_subdf.index = aggregated_subdf.index.astype(str)
    while (not pipedf.it.notnull().all()):
        # Rows are selected where to_node in last iteration leads to from node
        nextit=pipedf[pipedf['FROM_NODE'].isin(lastit['TO_NODE'])]
        # Since more than one pipe can lead to 1 node, rows are droped if cumq is nan
        # function cumq and cumq2, are set to only sum values if all values leading to node are NOT nan
        # so that this step works. 
        nextit = nextit.dropna(subset=["cumQ"])
        nextit.loc[:, 'it'] = it
        nextit=nextit.reset_index().set_index('FROM_NODE')
        
        
        # Now join the aggregated DataFrame with head
        if 'Ais' not in nextit.columns:
            nextit=nextit.join(aggregated_subdf[['Ais']])
            
        nextit.update(aggregated_subdf)
        nextit['Ais']=nextit['Ais'].fillna(0)
        nextit['qdw']=nextit['qdw'].fillna(0)
        nextit['qgw']=nextit['qgw'].fillna(0)
        # sqdw now sums cumulated qdw from each previous iteration with own qdw
        nextit['sqdw']=nextit.sqdw+nextit.qdw
        nextit['sqgw']=nextit.sqgw+nextit.qgw
        nextit['tfmin']=Tfmin
        nextit['z']=z 
        nextit['tf']=nextit.cuml/(Vest*60)
        nextit['tr']=nextit[['tfmin','tf']].max(axis=1)  
        nextit['rtrz']=r15*(38/(9+nextit.tr))*(nextit.z**0.25-0.369)
        nextit['Qmax']=nextit.rtrz*nextit.Ais
        nextit['cumQ']=nextit.cumQ+nextit.Qmax
        nextit['Qtot']=nextit.cumQ+nextit.sqdw+nextit.sqgw
        
        # print(it)
        # if it>1 :
        #     with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
        #         print(nextit[nextit['Qtot']==0])
        
        nextit[['D','sumtf','rtrz','Qmax','Qtot','okqdw']]=nextit.apply(lambda x:fund(x['Qtot'],x['Qmax'],x['sqdw'],x['slope'],x['length'],x['sumtf'],x['tf'],x['Ais'],x['z'],x['rtrz'],qfqtt,x['cumd'],Tfmin,r15),axis=1)
        # cumQ has to be recalculated , in case the value of Qmax and Qtot changed
        nextit['cumQ']=nextit.Qtot-nextit.sqdw 
        nextit=nextit.reset_index().set_index('LINK_ID')
        
        pipedf.update(nextit[['Qtot','cumQ','it','sqdw','qdw','qgw','sqgw','Qmax','sumtf','D','okqdw']],overwrite=True)
        # Function cumq2 is used here instead of cumq
        pipedf[['cumQ','sqdw','sqgw']]=pipedf.apply(lambda x:cumq2(x['cumQ'],x['Qmax'],x['qdw'],x['sqdw'],x['qgw'],x['sqgw'],x['FROM_NODE'],pipedf),axis=1)
        pipedf[['cuml','sumtf','cumd']]=pipedf.apply(lambda x:cuml(x['cuml'],x['length'],x['sumtf'],x['FROM_NODE'],pipedf,x['cumd']),axis=1)
        it=it+1
        lastit=nextit
        
    return pipedf
