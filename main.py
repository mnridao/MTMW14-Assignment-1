"""
MTMW14 Assignment 1 - ENSO

Student ID: 31827379
"""

import numpy as np

from ensoEquations import TemperatureEqn, HeightEqn, Parameters
from solver import Solver, Model
import helperFunctions as helpers
from stability import runStabilityAnalysis, plotSpectralRadius

## GLOBAL VARIABLES (just because jupyter notebooks) ##

T0 = 1.125                                         # Initial SST anomaly.
h0 = 0.                                            # Initial thermocline depth.

model = Model([TemperatureEqn(), HeightEqn()])     # Set up the problem.

scheme = helpers.setScheme("rk4")                  # Default scheme.

period = 41                                        # months
dt = 1/30                                          # 1 day
nt = int(period/dt)
solver = Solver(model, scheme, dt, nt, store=True)

dataScale = [7.5, 15, 2]                           # [Tscale, hscale, tscale]
solver.dimensionalise(True, dataScale)             # Dimensionalise the problem.

#%% TASKS 

def stabilityAnalysis():
    """ 
    Runs the stability analysis for Task A"""
    
    # Retrieve the default parameters of the problem.
    params = Parameters()
    
    # Set up analytical stability analysis and plot results.
    dts = np.arange(0, 10, 0.1)
    schemeNames = ["euler", "euler2", "trapezoidal", "heun", "rk4"]
    spectralRadius = runStabilityAnalysis(schemeNames, dts, params)
    plotSpectralRadius(spectralRadius, dts, schemeNames)
    
def runSolver(numPeriods, dt, en, mu, windForcing=False, annualCycling=False, 
              numEnsembles=1, pertRange=None):
    """ 
    A general function for running the solver when we aren't worried about 
    varying parameters within the function.
    
    Inputs
    -------
    numPeriods    : int
        Number of periods to run the simulation.
    dt            : float
        Time step
    en            : float
        Non-linearity parameter
    mu            : float
        Either mu (if annual cycling turned off) or mu_0 (if turned on), the 
        relative coupling parameter
    windForcing   : bool
        Turn on/off stochastic wind forcing
    annualCycling : bool
        Turn on/off annual cycling 
    numEnsembles  : int
        Number of ensembles to run (if running ensemble simulation)
    pertRange     : list of floats 
        Perturbation ranges of the initial conditions (if running ensemble)
    """
    
    solver.dt = dt
    solver.nt = int(period*numPeriods/solver.dt)  # Choose so that this is int.
    
    # Non-linearity.
    solver.model.setEn(en)
    
    # Set mu or annual cycling.
    if annualCycling:
        solver.model.setAnnualCycle(annualCycling)
        solver.model.setMu0(mu)
    else:
        solver.model.setMu(mu)
    
    # Set wind forcing
    solver.model.setWindForcing(windForcing, solver.dt, solver.nt)
    
    # Run the model
    if numEnsembles > 1:
        solver.runEnsemble(numEnsembles, pertRange, T0, h0)
    else:
        solver.run(T0, h0)
    
    # Plot the results.
    time = np.arange(0, solver.dt*(solver.nt + 1), solver.dt)
    if numEnsembles > 1:
        helpers.plotEnsembleTimeSeries(solver.ensembleHistory, time)
    else:
        helpers.plotTimeSeriesAndTrajectories(solver.history, time)

def runTaskB(numPeriods):
    """ 
    Runs the solver for different values of mu
    
    Inputs
    -------
    numPeriods : int
        number of periods to run the solver for
    """
    
    solver.nt = numPeriods*nt
    
    # Run and plot for the different values of mu.
    muValues = [2/3, 0.65, 0.66, 0.68]
    labels = ["$\mu$=2/3 (critical)"] + [f"$\mu$={mu}" for mu in muValues[1:]]
    cs = ['k--'] + ['-']*len(muValues[1:])
    helpers.runAndPlotVaryingMu(muValues, labels, cs, solver, T0, h0)
    
def runTaskCPart1(numPeriods):
    """ 
    Runs the solver for different values of en (mu = 2/3)
    
    Inputs
    -------
    numPeriods : int
        number of periods to run the solver for
    """
    
    solver.nt = numPeriods*nt
    
    # Set coupling parameter as critical value.
    solver.model.setMu(2/3)
    
    # Compare linearity to non-linearity.
    enValues = [0., 0.1, 0.5]
    
    data = []
    for en in enValues:
        solver.model.setEn(en)
        solver.run(T0, h0)
        data.append(np.row_stack(solver.history))
    
    # Plot results.
    time = np.arange(0, solver.dt*(solver.nt+1), solver.dt)
    labels = ["$e_n$=0 (linear)"] + [f"$e_n$={en}" for en in enValues]
    cs = ["k--"] + ["-"]*len(enValues[1:])    
    helpers.plotCombinedEnsemble(data, time, labels, cs)
    
def runTaskCPart2(numPeriods):
    """ 
    Runs the solver for different supercritical values of mu (en=0.1)
    
    Inputs
    -------
    numPeriods : int
        number of periods to run the solver for
    """    
    solver.nt = numPeriods*nt
    
    # Compare linearity to non-linearity.
    solver.model.setEn(0.1)
    muValues = [2/3, 0.68, 0.75]
    
    data = []
    for mu in muValues:
        solver.model.setMu(mu)
        solver.run(T0, h0)
        data.append(np.row_stack(solver.history))
    
    # Plot results.
    time = np.arange(0, solver.dt*(solver.nt+1), solver.dt)
    labels = ["$\mu$=2/3 (critical)"] + [f"$\mu$={mu}" for mu in muValues[1:]]
    cs = ["k--"] + ["-"]*len(muValues[1:])
    helpers.plotCombinedEnsemble(data, time, labels, cs)

def runTaskE(mu):
    """ 
    Test stochastic initiation hypothesis with noisy wind forcing and annual
    cycling (in the coupling parameter) turned off.
    
    Inputs
    -------
    mu : float
        coupling parameter
    """
    
    franValues = [0., 0.2]
    
    # Plot and run.
    helpers.runAndPlotTaskEPart1(franValues, mu, solver, T0, h0)
    
def runTaskEAnnualCycling(numPeriods, dt):
    """ 
    Test stochastic initiation with annual cycling.
    
    Inputs
    -------
    numPeriods : int
        Number of periods to run the simulation.
    dt         : float
        Time step.
    """
    
    solver.dt = dt
    solver.nt = int(period*numPeriods/solver.dt)
    time = np.arange(0, solver.dt*(solver.nt+1), solver.dt)  
    
    solver.model.setAnnualCycle(True)
    muValues = [2/3, 0.68, 0.69]
    
    data = []
    for mu0 in muValues:
        solver.model.setMu0(mu0)
        solver.run(T0, h0)
        data.append(np.row_stack(solver.history))
    
    labels = ["$\mu$=2/3 (critical)"] + [f"$\mu$={mu}" for mu in muValues[1:]]
    helpers.plotCombinedEnsemble(data, time, labels)
