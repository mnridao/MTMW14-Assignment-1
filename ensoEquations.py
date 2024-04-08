"""
MTMW14 Assignment 1 - ENSO

Student ID: 31827379
"""

from abc import ABC, abstractmethod
import numpy as np

class Parameters(object):
    """ 
    Class that stores the parameters for the ENSO problem. Also responsible 
    for updating any time dependent parameters on each time step.
    """
    
    def __init__(self):
        
        self.activateAnnualCycle = False 
        self.activateWindForcing = False
                
        ## DEFAULT PARAMETERS ##
        self.mu = 2/3           # Coupling coefficient
        self.b0 = 2.5           # High end value of coupling parameter
        self.gamma = .75        # Feedback of thermocline gradient on SST grad
        self.c = 1.             # Damping rate of SST anomolies
        self.r = .25            # Damping on upper ocean heat content
        self.alpha = .125       # Relates easterly wind stress to 
        
        self.en = 0.            # Degree of non-linearity of ocean heat content
        self.zeta1 = 0.         # Random wind stress forcing added to system
        self.zeta2 = 0.         # Random heating added to system
        
        self.updateb()          # Thermocline slope
        self.updateR()          # Bjerknes positive feedback
        
        ## SELF EXCITATION PARAMETERS ##
        self.muAnn = 0.2 
        self.mu0   = 0.75 
        self.tau   = 12
        
        ## WIND FORCING PARAMETERS ##
        self.fAnn  = 0.02
        self.fRan  = 0.2 
        self.tauCorr = 1/30  
        
        self.W = None
        
    def updateb(self):
        self.b = self.mu * self.b0
    
    def updateR(self):
        self.R = self.gamma * self.b - self.c 
    
    def updateMu(self, t):
        """ 
        Update the atmosphere-ocean relative coupling parameter.
        
        Inputs
        -------
        t : float
            time
        """
        
        # Return early if nothing to update.
        if not self.activateAnnualCycle: return
        
        # Allow coupling parameter to vary on annual cycle.
        self.mu = self.mu0 * (1 + self.muAnn * 
                              np.cos(2*np.pi*t / self.tau - 5*np.pi/6))
    
    def updateZeta1(self, dt, t):
        """ 
        Update the stochastic wind forcing term.
        
        Inputs
        -------
        dt : float
            time step
        t  : float
            time
        """
        
        # Return early if nothing to update.
        if not self.activateWindForcing: return
                        
        # Random forcing.
        W = self.W[int(np.floor(t/self.tauCorr))]
        
        # Add wind stress forcing parameterization.
        self.zeta1 = (self.fAnn*np.cos(2*np.pi*t/self.tau) + 
                      self.fRan * W * self.tauCorr/dt)
    
    def updateParams(self, t=None, dt=None):
        """ 
        Updates any time dependent parameters stored in the object.
        
        Inputs
        -------
        dt : float
            time step
        t  : float
            time
        """        

        if t != None:
            self.updateMu(t)
            
        if t != None and dt != None:
            self.updateZeta1(dt, t)
            
        self.updateb()
        self.updateR()
    
    def setMu(self, mu):
        """ Set the value of the coupling parameter"""
        
        if not self.activateAnnualCycle:
            self.mu = mu
            self.updateParams()
        
        else:
            self.activateAnnualCycling(False, mu)
    
    def activateAnnualCycling(self, activate, mu=None):
        """ Activate annual cycling for the coupling parameter
        
        Inputs
        -------
        activate : bool 
            turn annual cycling on/off 
        mu       : float
            if turning annual cycling off, can set mu here.
        """
        
        self.activateAnnualCycle = activate
        
        if activate:
            self.updateParams(0)
        
        else:
            self.mu = mu if mu else 2/3  # Restore default if mu not provided.
            self.updateParams()
    
    def activateWindForce(self, activate, dt, nt, seed, zeta1=None):
        """ Activate stochastic wind forcing
        
        Inputs
        ------- 
        activate : bool 
            turn annual cycling on/off 
        dt       : float
            time step 
        t        : flaot 
            time 
        seed    : int 
            random number seed (so that random number at each time step is the 
                                same for all equations in the problem)
        zeta1  : float 
           can overrite zeta1
        """
        
        self.activateWindForcing = activate
        
        if activate:
            np.random.seed(seed)
            self.W = np.random.uniform(size=int(dt*nt/self.tauCorr)+2) * 2 - 1
            self.updateParams(0, dt)
            
        else:
            self.zeta1 = zeta1 if zeta1 else 0. # Restore default if necessary.
            self.W = None
            self.updateParams()
    
class BaseEqn(ABC):
    """ 
    Parent class of any equation for the ENSO problem. All ENSO equations must 
    inherit this class and overrite functions.
    """
    
    def __init__(self):
        self.params = Parameters()
        
    def __call__(self, t, dt, Te, hw):
        """ 
        Make the equation object callable.
        
        Inputs
        ------
        t  : float
            time 
        dt : float 
            time step 
        Te : float 
            SST anomaly 
        hw : float 
            thermocline depth
        """
                        
        # Update all other parameters (in case of time dependent params).
        self.params.updateParams(t, dt)
        
        # Calculate RHS of differential eqn.
        return self._f(Te, hw)

    @abstractmethod 
    def bRowNonLinear(self, beta, dt, nt, *phiPs):
        pass

    @abstractmethod
    def bRow(self, beta, dt, nt):
        pass
    
    @abstractmethod 
    def aRow(self, beta, dt, nt):
        pass
    
    @abstractmethod 
    def _f(self):
        pass
        
class TemperatureEqn(BaseEqn):
    """ 
    SST anomaly equation for the ENSO problem.
    """
        
    def __init__(self):
        super().__init__()
    
    def bRowNonLinear(self, beta, dt, nt, *phiPs):
        """ 
        In the case of an implicit time scheme, the non-linear term must be 
        dealt with. Here we assume that a predictor step was used, such that 
        the non-linear terms can be moved to the right-hand-side of the
        implicit matrix equation (Ax = b).
        
        Inputs
        ------
        beta : float
            beta term for time scheme. 
        dt   : float 
            time step 
        nt   : int
            number of time iterations to run. 
        phiPs: the time step solutions to use for the non-linear term.
        """
        # Params evaluated at current time step.
        self.params.updateParams(dt*nt, dt)
        
        return -beta * dt * self.params.en * (phiPs[0] * self.params.b + 
                                              phiPs[1])**3
            
    def bRow(self, beta, dt, nt):
        """ 
        Called when an implicit time scheme is being used. Terms of the SST 
        equation that can be moved to the right-hand-side of the implicit 
        matrix equation.
        
        Inputs
        ------
        beta : float
            beta term for time scheme. 
        dt   : float 
            time step 
        nt   : int
            number of time iterations to run. 
        """                
        # Params evaluated at current time step.
        self.params.updateParams(dt*nt, dt)
        
        return beta * dt * (self.params.gamma * self.params.zeta1 + 
                            self.params.zeta2)
    
    def aRow(self, beta, dt, nt):
        """ 
        Called when an implicit time scheme is being used. Returns the 
        left-hand-side matrix of the implicit matrix equation (Ax=b).
        
        Inputs
        ------
        beta : float
            beta term for time scheme. 
        dt   : float 
            time step 
        nt   : int
            number of time iterations to run.
        """
        # Params evaluated at current time step.
        self.params.updateParams(dt*nt, dt)
        
        return [1 - beta * dt * self.params.R, 
                -beta * dt * self.params.gamma]
    
    def _f(self, Te, hw):
        """
        Calculate the right-hand-side of the SST differential equation.
        
        Inputs
        ------
        Te : float
            SST anomaly
        hw : float 
            thermocline depth
        """
        return (self.params.R*Te + self.params.gamma*hw - 
                self.params.en*(hw + self.params.b*Te)**3 + 
                self.params.gamma*self.params.zeta1 + self.params.zeta2)
    
class HeightEqn(BaseEqn):
    """ 
    Thermocline depth equation for the ENSO problem.
    """
        
    def __init__(self):
        super().__init__()
    
    def bRowNonLinear(self, beta, dt, nt, *phiPs):
        """ 
        This equation does not have any non-linear terms.
        """
            
        return 0. # No non-linear term.
        
    def bRow(self, beta, dt, nt):
        """ 
        Called when an implicit time scheme is being used. Terms of the hw 
        equation that can be moved to the right-hand-side of the implicit 
        matrix equation.
        
        Inputs
        ------
        beta : float
            beta term for time scheme. 
        dt   : float 
            time step 
        nt   : int
            number of time iterations to run. 
        """        
        return - beta * self.params.alpha * dt * self.params.zeta1
    
    def aRow(self, beta, dt, nt):
        """ 
        Called when an implicit time scheme is being used. Returns the 
        left-hand-side matrix of the implicit matrix equation (Ax=b).
        
        Inputs
        ------
        beta : float
            beta term for time scheme. 
        dt   : float 
            time step 
        nt   : int
            number of time iterations to run.
        """
        return [beta * self.params.alpha * self.params.b * dt,
                1 + beta * dt * self.params.r]
    
    def _f(self, Te, hw):
        """
        Calculate the right-hand-side of the thermocline differential equation.
        
        Inputs
        ------
        Te : float
            SST anomaly
        hw : float 
            thermocline depth
        """
        return (-self.params.r*hw - self.params.alpha*self.params.b*Te - 
                self.params.alpha*self.params.zeta1)
        