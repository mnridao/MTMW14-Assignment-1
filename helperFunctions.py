"""
MTMW14 Assignment 1 - ENSO

Student ID: 31827379
"""
import numpy as np
import matplotlib.pyplot as plt

import numericalSchemes as schemes
from matplotlib.gridspec import GridSpec

def setScheme(s):
    """ Selects the numerical scheme to use
    Inputs
    -------
    s : string or int
        The key for the numerical scheme that you want.
    """
    if s == 1 or s == "euler":
        scheme = schemes.forwardEulerSchemeCoupled
    elif s == 2 or s == "heun":
        scheme = schemes.heunSchemeCoupled
    elif s == 3 or s == "trapezoidal":
        scheme = schemes.trapezoidalSchemeCoupled 
    elif s == 4 or s == "rk4":
        scheme = schemes.RK4SchemeCoupled
    elif s == 5 or s == "euler2":
        scheme = schemes.forwardEulerImprovedSchemeCoupled
    elif s == 6 or s == "heun2":
        scheme = schemes.heunImprovedSchemeCoupled
    else: 
        return None
    return scheme

def plotTimeSeriesAndTrajectories(data, time):
    """ 
    Plots time series and trajectories on subplots. Plots are now very small, 
    and I did not have time to make them look nicer.
    
    Inputs
    -------
    data : np.array 
        solutions to the ENSO problem for SST anomaly and thermocline depth.
    time : np.array
        array of time in steps of dt for the simulation run.
    """
    
    T = data[:, 0]
    h = data[:, 1]
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), 
                                  gridspec_kw={'width_ratios' : [3, 2]})
    
    # Time series subplot.
    ax1.plot(time, T, label="$T_{E}$", linewidth=0.75)
    ax1.plot(time, h, label="$h_{w}$", linewidth=0.75)
    ax1.set_xlabel("Time [Months]", fontsize=10)
    ax1.set_ylabel("$T_{E}$ [K], $h_{w}$ [dm]", fontsize=10)
    ax1.set_xlim([time[0], time[-1]])
    ax1.tick_params(labelsize=10)
    ax1.grid()
    ax1.legend(fontsize=10, loc="best")
    
    # Trajectory subplot.
    ax2.plot(T, h, linewidth=0.75)
    ax2.plot(T[0], h[0], 'ok')  # Starting point.
    ax2.set_xlabel("$h_{w}$ [dm]", fontsize=10)
    ax2.set_ylabel("$T_{E}$ [K]", fontsize=10)
    ax2.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)
    ax2.grid()
    
    f.tight_layout()
    plt.show()

def plotCombinedEnsemble(ensembleData, time, labels=None, cs=None):
    """ 
    Plots time series and trajectories of ensemble on subplots.
    
    Inputs
    -------
    ensembleData : np.array 
        solutions to the ENSO problem for SST anomaly and thermocline depth.
    time         : np.array
        array of time in steps of dt for the simulation run.
    labels       : list of strings
        list containing labels for the plot corresponding to each ensemble 
        member.
    cs           : string
        plot colour, linestyle
    """
    # Setup figures.
    plt.figure(figsize=(15, 5))
    gs = GridSpec(2, 2, width_ratios=[3, 2], wspace=0.3)
    
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1])    
    
    # Plot time series on the left
    for i, ensemble in enumerate(ensembleData):
        
        T = ensemble[:, 0]
        h = ensemble[:, 1]
        
        label = labels[i] if labels else ""
        c = cs[i] if cs else '-'
        
        ax1.plot(time, T, c, label=label, linewidth=0.75)
        ax2.plot(time, h, c, label=label, linewidth=0.75)
    
        ax3.plot(T[0], h[0], 'ko')
        ax3.plot(T, h, c, label=label, linewidth=0.75)
    
    ax1.set_ylabel("$T_{E}$ [K]", fontsize=15)
    ax1.set_xlim([time[0], time[-1]])
    ax1.tick_params(labelsize=15)
    ax1.grid()
    
    ax2.set_ylabel("$h_{w}$ [dm]", fontsize=15)
    ax2.set_xlabel("Time [Months]", fontsize=15)
    ax2.set_xlim([time[0], time[-1]])
    ax2.tick_params(labelsize=15)
    ax2.grid()
    
    ax3.set_xlabel("$h_{w}$ [dm]", fontsize=15)
    ax3.set_ylabel("$T_{E}$ [K]", fontsize=15)
    ax3.legend(fontsize=15, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax3.tick_params(labelsize=15)
    ax3.grid()
    
    plt.show()

def plotEnsembleTimeSeries(ensembleData, time, labels=None, cs=None):
    """ 
    Only plot time series of ensemble on subplots.
    
    Inputs
    -------
    ensembleData : np.array 
        solutions to the ENSO problem for SST anomaly and thermocline depth.
    time         : np.array
        array of time in steps of dt for the simulation run.
    labels       : list of strings
        list containing labels for the plot corresponding to each ensemble 
        member.
    cs           : string
        plot colour, linestyle
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    for i, ensemble in enumerate(ensembleData):
        
        T = ensemble[:, 0]
        h = ensemble[:, 1]
        
        label = labels[i] if labels else ""
        c = cs[i] if cs else '-'
        ax1.plot(time, T, c, label=label, linewidth=0.5)
        ax2.plot(time, h, c, label=label, linewidth=0.5)
    
    ax1.set_ylabel("$T_{E}$ [K]", fontsize=15)
    ax1.set_xlabel("Time [Months]", fontsize=15)
    ax1.set_xlim([time[0], time[-1]])
    ax1.tick_params(labelsize=10)
    ax1.grid()
     
    ax2.set_ylabel("$h_{w}$ [dm]", fontsize=15)
    ax2.set_xlabel("Time [Months]", fontsize=15)
    ax2.set_xlim([time[0], time[-1]])
    if labels:
        ax2.legend(fontsize=10, loc="best")
    ax2.tick_params(labelsize=10)
    ax2.grid()
    
    fig.tight_layout()
    plt.show()

def runAndPlotTaskEPart1(franValues, mu, solver, *phi0s):
    """ 
    Runs and plots the first part of Task E.
    
    Inputs
    -------
    franValues : list of floats
        Values of f_{ran} that will be run through the solver and plotted 
        together
    mu         : float
        Coupling parameter (fixed)
    solver     : Solver object
        ENSO solver
    phi0s      : floats
        Initial conditions.
    """
    
    solver.dt = 1/30
    solver.nt = int(41*15 / solver.dt)
    time = np.arange(0, solver.dt*(solver.nt + 1), solver.dt)
    
    solver.model.setEn(0.)
    solver.model.setAnnualCycle(False)
    solver.model.setWindForcing(True, solver.dt, solver.nt)
    solver.model.setFann(0.02)
    solver.model.setMu(mu)
    
    # Set up axes.
    plt.figure(figsize=(15, 5))
    gs = GridSpec(2, 2, width_ratios=[3, 2], wspace=0.3)
    
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1])  
    
    for fran in franValues:
        solver.model.setFran(fran)
        solver.run(*phi0s)
        
        T = solver.history[:, 0]
        h = solver.history[:, 1]
        
        label = f"$\Delta t$={solver.dt:.2f}, " + "$f_{ran}$=" + f"{fran:.2f}"
        ax1.plot(time, T, label=label, linewidth=0.5)
        ax2.plot(time, h, label=label, linewidth=0.5)
        
        ax3.plot(T, h, label=label, linewidth=0.5)
        ax3.plot(T[0], h[0], 'ko')
    
    # Run for half the time step.
    solver.dt = 1/60
    solver.nt = int(41*15 / solver.dt)
    time = np.arange(0, solver.dt*(solver.nt + 1), solver.dt)
    solver.run(*phi0s)
    
    T = solver.history[:, 0]
    h = solver.history[:, 1]
    
    label = f"$\Delta t$={solver.dt:.2f}, " + "$f_{ran}$=" + f"{fran:.2f}"
    ax1.plot(time, T, label=label)
    ax2.plot(time, h, label=label)
    
    ax3.plot(T, h, label=label)
    ax3.plot(T[0], h[0], 'ko')
    
    ax1.set_ylabel("$T_{E}$ [K]", fontsize=15)
    ax1.set_xlim([time[0], time[-1]])
    ax1.tick_params(labelsize=15)
    ax1.grid()
     
    ax2.set_ylabel("$h_{w}$ [dm]", fontsize=15)
    ax2.set_xlabel("Time [Months]", fontsize=15)
    ax2.set_xlim([time[0], time[-1]])
    ax2.tick_params(labelsize=10)
    ax2.grid()
            
    ax3.set_ylabel("$h_{w}$ [dm]", fontsize=15)
    ax3.set_xlabel("$T_{E}$ [K]", fontsize=15)
    ax3.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax2.tick_params(labelsize=10)
    ax3.grid()
    plt.show()
    
def runAndPlotVaryingMu(muValues, labels, cs, solver, *phi0s):
    """
    Used in Task B - the solver is run for different values of mu.
    
    Parameters
    ----------
    muValues : list
        Values of mu that will be run.
    solver : Solver object
        class that runs the ENSO model.
    *phi0s : float
        Initial conditions for the solver.
    """
    
    time = np.arange(0, solver.dt * (solver.nt + 1), solver.dt)
    
    # Setup figures.
    plt.figure(figsize=(15, 5))
    gs = GridSpec(2, 2, width_ratios=[3, 2], wspace=0.3)
    
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1])
    
    for i, mu in enumerate(muValues):
        
        solver.model.setMu(mu)
        solver.run(*phi0s)
    
        T = solver.history[:, 0]
        h = solver.history[:, 1]
    
        ax1.plot(time, T, cs[i], label=labels[i], linewidth=0.75)
        ax2.plot(time, h, cs[i], label=labels[i], linewidth=0.75)
    
        ax3.plot(T[0], h[0], 'ko')
        ax3.plot(T, h, cs[i], label=labels[i], linewidth=0.75)
        
    ax1.set_ylabel("$T_{E}$ [K]", fontsize=15)
    ax1.set_xlim([time[0], time[-1]])
    ax1.tick_params(labelsize=15)
    ax1.grid()
    
    ax2.set_ylabel("$h_{w}$ [dm]", fontsize=15)
    ax2.set_xlabel("Time [Months]", fontsize=15)
    ax2.set_xlim([time[0], time[-1]])
    ax2.tick_params(labelsize=15)
    ax2.grid()
    
    ax3.set_xlabel("$h_{w}$ [dm]", fontsize=15)
    ax3.set_ylabel("$T_{E}$ [K]", fontsize=15)
    ax3.legend(fontsize=15, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax3.tick_params(labelsize=15)
    ax3.grid()
    
    plt.show()
     
def calculatePeriod(signal, dt):
    """ 
    Calculate the period of the oscillating signal with FFT.
    
    Inputs
    -------
    signal : np.array
        signal dataset
    dt    : float
        time step
    """
 
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=dt)
    print(fft_freq)
    
    # Find the dominant frequency
    dominant_frequency = np.abs(fft_freq[np.argmax(np.abs(fft_result))])
    
    # Period is the reciprocal of the frequency.    
    return 1 / dominant_frequency