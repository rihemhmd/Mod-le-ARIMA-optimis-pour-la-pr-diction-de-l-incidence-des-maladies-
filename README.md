# Mod-le-ARIMA-optimis-pour-la-pr-diction-de-l-incidence-des-maladies-
from random import randint

import warnings

from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

import random
import numpy as np
from math import sqrt,exp,cos,pi
import sklearn.metrics as metrics 


from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA


root= tk.Tk()
root.title("My Application")
canvas1 = tk.Canvas(root, width = 600, height = 400,bg='red')
canvas1.pack()
 
label1 = tk.Label(root, text='Prevision par ARIMA Optimisee',fg='black')
label1.config(font=('Arial', 20) )

canvas1.create_window(400, 50, window=label1)
 
 
def getCSV():
      global df,x,y,xtrain,xtest,ytrain,ytest
      f_types = [('CSV files',"*.csv")]
      import_file_path = filedialog.askopenfilename(filetypes=f_types)
     
      df = pd.read_csv (import_file_path)
      x = df.iloc[:,1]
      y = df.iloc[:,2]
      taux =int(len(x)*0.8)
      xtrain = x[:taux]
      xtest = x[taux:]
      ytrain = y[:taux]
      ytest = y[taux:]
      
def display_data():
     # Creating Figure.
     fig = Figure(figsize = (10,8), dpi = 100)
     # Plotting the graph inside the Figure
     
     plt.plot(x,y )
      
     plt.xlabel("Mois")
     plt.ylabel("incidence ")
     plt.title("graphe des incidences")
     plt.show()



#--- MAIN ---------------------------------------------------------------------+
def cost_func(pq):
     
    p,d,q = int(pq[0]),2,int(pq[1])

    model = ARIMA(ytrain, order = (0, 0, 0))  
    fitted = model.fit()  
    forecasted = fitted.forecast(len(ytest)) 
    mse = mean_squared_error(ytest,forecasted)
    rmse = sqrt(mse)
    aic = fitted.aic
    
    return aic

def DE(cost_func, popsize, mutate, recombination, maxiter):
    
    #--- INITIALIZE A POPULATION (step #1) ----------------+
    
    population = []
    for i in range(0,popsize):
        indv = []

        indv.append(random.randint(1,4))
        indv.append(random.randint(5,15))

        population.append(indv)
            
    GbestSol = [0, 2, 0] # best indiv in the population
    Gbest = 1000000.00  # a high value for best indiv in the population

    #--- SOLVE --------------------------------------------+
    
     
    # cycle through each generation (step #2)
    for iter in range(1,maxiter+1):
         
        # cycle through each individual in the population
        for j in range(0, popsize):

            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = [x for x in range(popsize)]
            canidates.remove(j)
            random_index = random.sample(canidates, 3)

            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]     # target individual
            #+-----------------------------+ 
            # adaptation du facteur alpha
            rd1 = random.random()
            term  = ((1-iter)/maxiter)**2
            alpha = 1-rd1**term  

            # adaptation du facteur F
            F = mutate[0]+(mutate[1]-mutate[0])*alpha
            #+-----------------------------+
            v = random.randint(2,8) 

            v_donorp = alpha*x_1[0] + F * v*abs(x_2[0]-x_3[0]) 
          
            v_donorq = alpha*x_1[1] + F * v*abs(x_2[1]-x_3[1]) 

            print('patienter, ----warning pas genant-----') 

            if v_donorp > 15:
               v_donorp = random.randint(1,15) 
            if v_donorq > 15:
               v_donorq = random.randint(1,15) 
            if v_donorp < 0:
               v_donorp = 0 
            if v_donorq < 0:
               v_donorq = 0

            #--- RECOMBINATION (step #3.B) ----------------+
            #+-----------------------------+
            #+------ adaptation de CR -----+
            
            CR = recombination[0]+alpha*(recombination[1]-recombination[0]) 
            crossover = random.random()
            if crossover <= CR:
                v_trialp = v_donorp 
                v_trialq = x_t[1] 
                     
            else:
                v_trialp = x_t[0] 
                v_trialq = v_donorq 
                     
            #-----------SELECTION (step #3.C) -------------+
            
            v_trialC = [int(v_trialp),int(v_trialq)]
            score_trialC = cost_func(v_trialC)
            score_target = cost_func(x_t)

            #---- on prend le v_trial s'il est meilleur -----+
            if score_trialC <= score_target  :
                population[j] = v_trialC 
                 
                if score_trialC < Gbest:
                   Gbest = score_trialC # fitness of the best indiv in the population
                   GbestSol = v_trialC # best indiv in the population
                   

            else:
                if score_target < Gbest:
                   Gbest = score_target # fitness of the best indiv in the population
                   
                   GbestSol = x_t # best indiv in the population
 
    p,d,q = GbestSol[0],2,GbestSol[1]
    print ("solution optimale: %d,%d,%d , fitness: %.5f" %(p,d,q,Gbest))
    print ("CLIQUER SUR : Graphe de Prevision" )
    return GbestSol 



def arima():
     global forecasted 
     #--- RUN ----------------------------------------------------------------------+
     d = 2 # dimension of the input vector; x1,x2,.....
 
     popsize = 50                       # taille de la Population 
     mutate = [0.1, 0.6]                        # Mutation factor [0,2]
     recombination = [0.1,0.4]                 # Recombination rate [0,1]
     maxiter = 2    
     sol = DE(cost_func, popsize, mutate, recombination, maxiter)
 
     p,d,q = sol[0], d ,sol[1]
     print('+---------------------------------------+')
       
     # Forecast
     model = ARIMA(ytrain, order=(p, d, q))  
     fitted = model.fit()
     forecasted = fitted.forecast(len(ytest))   
     return

def dispGraphPrev():
     # Creating Figure.
     fig = Figure(figsize = (10,8), dpi = 100)
     # Plotting the graph inside the Figure
     
     plt.plot(xtest,ytest, label='donnees de test' )
     plt.plot(xtest,forecasted, label='previsions de test'  )
     plt.legend()
     plt.xlabel("Mois")
     plt.ylabel("incidence")
     plt.title("prevision")
     plt.show()



browseButton_CSV = tk.Button(text='Charger le fichier CSV', command=getCSV, bg='black', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(220, 140, window=browseButton_CSV)

button2 = tk.Button (root,   text='Graphe des donnees    ', command=display_data, bg='black', fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(420, 140, window=button2) 

button3 = tk.Button (root,   text='Prevision avec ARIMA  ', command=arima, bg='black', fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(300, 180, window=button3)

button4 = tk.Button (root,   text='Graphe de Prevision  ', command=dispGraphPrev, bg='black', fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(300, 220, window=button4)

 
button5 = tk.Button (root, text='Exit!', command=root.destroy, bg='black', fg='white',font=('helvetica', 11, 'bold'))
canvas1.create_window(300, 280, window=button5)
 
root.mainloop()






