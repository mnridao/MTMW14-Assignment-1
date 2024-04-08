"""
MTMW14 Assignment 1 - ENSO

Student ID: 31827379
"""

import numpy as np

class Solver(object):
    """ Class that runs the ENSO simulaton."""
    
    def __init__(self, model, scheme, dt, nt, store=False):
        """ 
        model  : Model object
            class that contains the equations for the ENSO problem
        scheme : callable object
            the time scheme used in the simulation.
        dt     : float
            time step 
        nt     : int 
            number of time steps to run in the simulation 
        store  : bool 
            turn storage on/off
        """
        
        self.isDimensionalised = False
        self.scaleData = None
        
        self.model   = model
        self.scheme  = scheme
        self.dt      = dt
        self.nt      = nt
        
        self.store   = store
        self.history = None
        self.ensembleHistory = None
            
    def dimensionalise(self, isDimensionalised, scaleData=None):
        """ 
        Dimensionalise the parameters of the problem.
        
        Inputs
        ------
        isDimensionalised : bool
            turn dimensionalisation on/off 
        scaleData         : list of floats
            list of values to dimensionalise variables with. Order is in the 
            order of initial conditions and last index is for time scale, 
            i.e. [Tscale, hscale, tscale]
        """
        self.isDimensionalised = isDimensionalised
        self.scaleData = scaleData
        
        self.model.scaleData(isDimensionalised, scaleData)
    
    def run(self, *phi0):
        """ 
        Run the solver.
        
        Inputs
        ------
        phi0 : float
           initial conditions for the problem.
        """
                              
        # Initialise storage arrays if necessary.
        if self.store:
            self.history = np.zeros(shape=(self.nt+1, len(self.model.eqns)))
            self.history[0, :] = phi0
        
        # Scale initial conditions if necessary.
        if self.isDimensionalised:
            phi0 = np.array(phi0) / np.array(self.scaleData)[:-1]
        
        for i in range(1, self.nt+1):
            
            dt = self.dt / (self.scaleData[-1] if self.isDimensionalised 
                                               else 1.)
            
            # Calculate new time step values.
            phi = self.scheme(self.model.eqns, dt, i, *phi0)
            
            # Update previous time step values.
            phi0 = phi
            
            # Store results if necessary.
            if self.model.isScaled:
                phi = phi * np.array(self.model.dataScale)[:-1]
            
            if self.store:
                self.history[i, :] = phi
    
        return phi
    
    def runEnsemble(self, numEnsembles, perturbationRange, *phi0):
        """ 
        Run the solver for an ensemble simulation.
        
        Inputs
        ------
        numEnsembles : int
            number of ensembles to run
        perturbationRange : list of floats
            range at which to perturb the initial conditions for each run
        phi0 : float
           initial conditions for the problem.
        """
        
        # Copy - find better fix :(
        phi0 = phi0[:]
        
        # Overwrite store state if necessary (should be True in ensemble).
        store = self.store
        self.store = True
        
        # Initialise storage.
        self.ensembleHistory = []       # List for now.
        
        for _ in range(numEnsembles):
            
            # Non dimensionalise perturbation range.
            pert = np.array(perturbationRange) / np.array(self.scaleData)[:-1]
            
            # Perturb initial condition.
            phiPert = (np.array(phi0) + pert * (np.random.uniform(size=2) 
                                                * 2 - 1))
            
            # Run model.
            _ = self.run(*phiPert)
            
            # Store run.
            self.ensembleHistory.append(self.history)
        
        # Restore previous store state.
        self.store = store
    
class Model(object):
    """ 
    Class responsible for holding the equations to the ENSO problem, and
    updating their parameters if they are manually set.
    """
    
    def __init__(self, eqns):
        """ 
        Inputs
        -------
        eqns : list of BaseEqn objects
            Each equation that appears in the ENSO problem.
        """
        
        self.eqns = eqns
        
        self.isScaled    = False
        self.annualCycle = False
        self.windForcing = False
        
        self.dataScale = None
    
    def scaleData(self, isScaled, dataScale):
        """ 
        Scales the data.
        
        Inputs
        -------
        isScaled : bool
            turn scaling on/off 
        dataScale : list of floats
            list containing the scalings in the order [Tscale, hscale, tscale]
        """
        self.isScaled = isScaled
        self.dataScale = dataScale if isScaled else None
        
        # Scale time parameters as well.
        for func in self.eqns:
            func.params.tau *= 1/dataScale[-1] if isScaled else dataScale[-1]
            func.params.tauCorr *= 1/dataScale[-1] if isScaled else dataScale[-1]
        
    def setMu(self, mu):
        """ 
        Set the atmosphere-ocean coupling parameter
        
        Inputs
        -------
        mu : float
            coupling parameter
        """
        # Update mu for each of the equations in the model.
        for func in self.eqns:
            func.params.setMu(mu)
    
    def setMu0(self, mu0):
        """ 
        Set the atmosphere-ocean coupling parameter mu0
        
        Inputs
        -------
        mu0 : float
            coupling parameter
        """
        # Update mu0 for each of the equations in the model.
        for func in self.eqns:
            func.params.mu0 = mu0
            func.params.updateParams() # Does this do anything?
    
    def setEn(self, en):
        """ 
        Set the non-linearity parameter en
        
        Inputs
        -------
        en : float
            non-linearity parameter
        """
        # Update en for each of the equations in the model.
        for func in self.eqns:
            func.params.en = en
            func.params.updateParams()
    
    def setFann(self, fann):
        """ 
        Set the wind forcing seasonal cycling"""
        # Update fann for each of the equations in the model.
        for func in self.eqns:
            func.params.fAnn = fann 
            func.params.updateParams()
            
    def setFran(self, fran):
        """ 
        Set the wind forcing stochastic term"""
        # Update fran for each of the equations in the model.
        for func in self.eqns:
            func.params.fRan = fran 
            func.params.updateParams()
    
    def setAnnualCycle(self, activate):
        """ 
        Activate the coupling parameter annual cycling
        
        Inputs
        -------
        activate : bool
            turn annual cyling on/off
        """
        for func in self.eqns:
            func.params.activateAnnualCycling(activate)
            
    def setWindForcing(self, activate, dt=None, nt=None):
        """ 
        Activate the wind forcing
        
        Inputs
        -------
        activate : bool
            turn wind forcing on/off 
        dt  : float
            time step 
        nt  : int 
          number of time steps to run in simulation (needed for W array)"""
        # Check for this nonsense input :(.
        if activate and not dt and not nt: return
        
        # Seed for W should be the same for each of the equations.
        seed = np.random.randint(0, 1e6)
        
        for func in self.eqns:
            func.params.activateWindForce(activate, dt, nt, seed)