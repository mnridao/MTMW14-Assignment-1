"""
MTMW14 Assignment 1 - ENSO

Student ID: 31827379
"""

import numpy as np

def forwardEulerSchemeCoupled(funcs, dt, nt, *phi0s):
    """
    Forward euler scheme for coupled equations (first order accurate)
    
    Inputs
    -------
    funcs  : list of callable objects
             RHS of differenetial equation to be solved numerically.
    dt     : float
             Time step.
    *phi0s : 
             Previous time step parameters for funcs (all funcs must have
             same arguments for this to work).    
    """   
        
    # Check that funcs and phi0s are same length.
    if len(funcs) != len(phi0s):
        return None
        
    # Iterate through series of coupled equations.
    phi = np.zeros(shape=len(funcs))
    for i, (func, phi0) in enumerate(zip(funcs, phi0s)):
        
        # Forward euler.
        phi[i] = phi0 + dt * func(dt*(nt - 1), dt, *phi0s)
    
    return phi

def forwardEulerImprovedSchemeCoupled(funcs, dt, nt, *phi0s):
    """
    Forward euler improved scheme (Matsuno) for coupled equations 
    (first order accurate)
    
    Inputs
    -------
    funcs  : list of callable objects
             RHS of differenetial equation to be solved numerically.
    dt     : float
             Time step.
    *phi0s : 
             Previous time step parameters for funcs (all funcs must have
             same arguments for this to work).    
    """   
    
    # Check that funcs and phi0s are same length.
    if len(funcs) != len(phi0s):
        return None
        
    # Copy initial condition.
    phi0s = phi0s[:]
    
    # Iterate through series of coupled equations.
    phi = np.zeros(shape=len(funcs))
    for i, (func, phi0) in enumerate(zip(funcs, phi0s)):
        
        # Forward euler.
        phi[i] = phi0 + dt * func(dt*(nt - 1), dt, *phi0s)
        
        # Replace ith initial condition with last guess.
        phi0s = (*phi0s[:i], phi[i], *phi0s[i+1:])
        
    return phi

def heunSchemeCoupled(funcs, dt, nt, *phi0s):
    """
    Heun scheme for coupled equations (second order accurate)
    
    Inputs
    -------
    funcs  : list of callable objects
             RHS of differenetial equation to be solved numerically.
    dt     : float
             Time step.
    *phi0s : 
             Previous time step parameters for funcs (all funcs must have
             same arguments for this to work).    
    """   
    
    # Check that funcs and phi0s are same length.
    if len(funcs) != len(phi0s):
        return None
    
    # Euler Predicter step.
    phiPs = forwardEulerSchemeCoupled(funcs, dt, nt, *phi0s)
    
    # Iterate through series of coupled equations.
    phi = np.zeros(shape=len(funcs))
    for i, (func, phi0) in enumerate(zip(funcs, phi0s)):
        
        # Heun corrector step.
        phi[i] = phi0 + 0.5 * dt * (func(dt*(nt-1), dt, *phi0s) + 
                                    func(dt*nt, dt, *phiPs))
    
    return phi

def heunImprovedSchemeCoupled(funcs, dt, nt, *phi0s):
    """
    Heun 2 scheme with predictor step in between. Did not end up using this.
    """   
    # Check that funcs and phi0s are same length.
    if len(funcs) != len(phi0s):
        return None
    
    # Euler Predicter step.
    phiPs = forwardEulerImprovedSchemeCoupled(funcs, dt, nt, *phi0s)
    
    # Iterate through series of coupled equations.
    phi = np.zeros(shape=len(funcs))
    for i, (func, phi0) in reversed(list(enumerate(zip(funcs, phi0s)))):
        
        # Heun corrector step.
        phi[i] = phi0 + 0.5 * dt * (func(dt*(nt-1), dt, *phi0s) + 
                                    func(dt*nt, dt, *phiPs))
        
        # Replace ith predictor with last guess.
        phiPs = (*phiPs[:i], phi[i], *phiPs[i+1:])
    
    return phi
    
def RK4SchemeCoupled(funcs, dt, nt, *phi0s):
    """ 
    Runge-Kutta 4th order scheme for coupled equations.
    
    Inputs
    -------
    funcs  : list of callable objects
             RHS of differenetial equation to be solved numerically.
    dt     : float
             Time step.
    *phi0s : 
             Previous time step parameters for funcs (all funcs must have
             same arguments for this to work).  
    """
    
    # Initialise array of k values (1 - 4 for RK4).
    k = np.zeros(shape=(4, len(funcs)))
    
    phiPs = phi0s
    nti = nt - 1
    for i in range(4):
        for j, func in enumerate(funcs):
            
            # Update k value.
            k[i, j] = func(nti*dt, dt, *phiPs)
            
        # Update RK prediction value for RK4.
        phiPs = phi0s + dt * k[i, :] * (0.5 if i < 2 else 1)
        nti = nt - 0.5 if i < 2 else 0
        
    # Calculate new timestep.
    phi = np.zeros(shape=len(funcs))
    for i, (func, phi0) in enumerate(zip(funcs, phi0s)):
        
        # Runge-Kutta step.
        phi[i] = phi0 + dt * (k[0, i] + 2*k[1, i] + 2*k[2, i] + k[3, i]) / 6
    
    return phi
    
def trapezoidalSchemeCoupled(funcs, dt, nt, *phi0s):
    """ 
    Trapezoidal (implicit) scheme for coupled equations.
    
    Inputs
    -------
    funcs  : list of callable objects
             RHS of differenetial equation to be solved numerically.
    dt     : float
             Time step.
    *phi0s : 
             Previous time step parameters for funcs (all funcs must have
             same arguments for this to work).  
    """
    
    # Check that funcs and phi0s are same length.
    if len(funcs) != len(phi0s):
        return None
    
    # Forward step prediction for non-linear term.
    phiPs = forwardEulerSchemeCoupled(funcs, dt, nt, *phi0s)
    
    # Setup matrix equation to solve for current time step.
    a = np.zeros(shape=(len(funcs), len(funcs)))
    b = np.zeros(shape=len(funcs))
    for i, (func, phi0) in enumerate(zip(funcs, phi0s)):
        a[i] = np.array(func.aRow(0.5, dt, nt))
        b[i] = np.array(phi0 + func.bRow(0.5, dt, nt) + 
                               func.bRowNonLinear(0.5, dt, nt, *phiPs) + 
                               0.5 * dt * func(dt*(nt-1), dt, *phi0s))
    
    # Solve matrix equation for current time step.
    return np.linalg.solve(a, b).tolist()