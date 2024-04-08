"""
MTMW14 Assignment 1 - ENSO

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt

def amplificationMatrix(scheme, params, dt):
    """ 
    Return the amplification matrix of a scheme.
    
    Inputs
    ------
    scheme : string or int
        The scheme that you want the amplification matrix for.
    params : Parameters object
        Object containing all the parameters relevent to the problem.
    dt    : float
        time step.
    
    Returns
    -------
    np.array
        The amplification matrix.
    """
    
    A = np.array([[-params.r, -params.alpha * params.b], 
                  [params.gamma, params.R]])
    
    if scheme == "euler":
        return np.eye(2) + dt * A
    
    elif scheme == "euler2":
        L1 = np.eye(2)
        L1[1, 0] = - dt * params.gamma
        
        L2 = np.eye(2) + dt * A 
        L2[1, 0] = 0.
        
        return np.matmul(np.linalg.inv(L1), L2)
    
    elif scheme == "trapezoidal":
        L1 = np.eye(2) - 0.5 * dt * A
        L2 = np.eye(2) + 0.5 * dt * A
        return np.matmul(np.linalg.inv(L1), L2)
    
    elif scheme == "heun":
        Leuler = amplificationMatrix("euler", params, dt)
        return np.eye(2) + 0.5 * dt * np.matmul(A, Leuler + np.eye(2))
    
    elif scheme == "heun2":
        Leuler2 = amplificationMatrix("euler2", params, dt)
        
        L1 = np.eye(2)
        L1[0, 1] = -0.5 * params.alpha * params.b * dt
        
        L2 = A
        L2[0, 1] = 0.
        
        return np.matmul(np.linalg.inv(L1), np.eye(2) + 0.5 * dt * A + 
                         0.5 * dt * np.matmul(L2, Leuler2))
    
    elif scheme == "rk4":
        L1 = A
        L2 = np.matmul(A, np.eye(2) + 0.5 * dt * L1)
        L3 = np.matmul(A, np.eye(2) + 0.5 * dt * L2)
        L4 = np.matmul(A, np.eye(2) + dt * L3)
        
        return np.eye(2) + dt / 6 * (L1 + 2*L2 + 2*L3 + L4)
    
    return None
    
def runStabilityAnalysis(schemes, dts, params):
    """ 
    Run stability analysis.
    
    Inputs
    ------
    schemes : list of string/ints
        The schemes that you want to perform stability analysis for.
    params  : Parameters object
        Object containing all the parameters relevent to the problem.
    dts     : np.array
        array of time steps.
    
    Returns
    -------
    np.array
        array of spectral radius for each scheme.
    """
    eigsSchemes = []
    for scheme in schemes:
        eigs = []
        
        for dt in dts:
            
            L = amplificationMatrix(scheme, params, dt)
            eig, _ = np.linalg.eig(L)
            eigs.append(np.abs(np.max(eig)))
            
        eigsSchemes.append(eigs) 
        
    return np.array(eigsSchemes)

def plotSpectralRadius(spectralRadius, dts, schemes):
    """ 
    Plots the spectral radius for specific numerical schemes.
    
    Inputs
    ------
    spectralRadius  : array of floats
        Array of the spectral radii for each scheme in schemes
    dts             : np.array
        array of time steps.
    schemes         : list of string/ints
        The schemes that you want to perform stability analysis for.
    """
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), 
                            gridspec_kw={'width_ratios': [1, 1, 1]})
    
    # Plot on the left (empty)
    axs[0].axis("off")
    
    # Main plot in the center
    for i, pho in enumerate(spectralRadius):
        axs[1].plot(dts, pho, label=schemes[i])
    
    axs[1].set_xlabel("$\Delta$t", fontsize=15)
    axs[1].set_ylabel("Spectral radius", fontsize=15)
    axs[1].set_xlim([dts[0], dts[-1]+0.1])
    axs[1].tick_params(axis='both', which='both', labelsize=10)
    axs[1].legend(fontsize=10)
    axs[1].grid()
    
    # Plot on the right (empty)
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()