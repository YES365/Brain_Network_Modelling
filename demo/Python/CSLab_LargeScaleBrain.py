import numpy as np
import scipy
import cma

def Stuart_Landau_model(regional,simula_time,dt,A,SC,W,G,beta):
    # timing
    t_0 = 0
    t_steps = int((simula_time - t_0) / dt + 1)
    dsig = np.sqrt(dt) * beta  # precalculated timestep for noise
    X = np.zeros((regional, t_steps))
    Y = np.zeros((regional, t_steps))
    for t in range(1,t_steps):
        Diff = np.tile(X[:,t-1],(regional,1))-np.transpose(np.tile(X[:,t-1],(regional,1)))
        np.random.seed(t)
        n1=np.random.randn(regional)
        X[:,t] = X[:,t-1] + dt*((A-pow(X[:,t-1],2)-pow(Y[:,t-1],2))*X[:,t-1]-W.T*Y[:,t-1] + G*(np.sum(SC*Diff,axis=1))) + dsig*n1
        Diff = np.tile(Y[:,t-1],(regional,1))-np.transpose(np.tile(Y[:,t-1],(regional,1)))
        np.random.seed(t-1)
        n2=np.random.randn(regional)
        Y[:,t] = Y[:,t-1] + dt*((A-pow(X[:,t-1],2)-pow(Y[:,t-1],2))*Y[:,t-1]+W.T*X[:,t-1] + G*(np.sum(SC*Diff,axis=1))) + dsig*n2
        Signal=X
    return Signal

def Linear_gaussian_model(fitparameter,SC):
    ## simplified form: X=sqrt(2)*sigma/(A+GH)*epilson
    ##Laplace matrix H
    beta = fitparameter['beta'] ##inverse of the relaxation time
    G = fitparameter['G']## global conple strength
    a = fitparameter['a']##
    N_ROI=360
    SC=np.power(SC,beta)
    H=np.diag(np.sum(SC,axis=1))-SC
    SE,SEC=np.linalg.eig(H)
    H=H/max(SE.real)
    Q=np.sqrt(2)*np.linalg.inv(np.identity(N_ROI)*a+G*H)
    Cov=Q @ np.transpose(Q)
    D = np.diag(np.sqrt(np.diag(Cov)))
    D_inv = np.linalg.inv(D)
    FC_model=D_inv @ Cov @ D_inv
    return FC_model

def NMDA_neural_mess_model(fitparameter,SC):
    # A large-scale model for individual, return model-based BOLD signal.
    # Simulation Parameters, using Euler–Maruyama method for SDE.
    dt = 0.01  # s
    Simulation_During = 16.4 * 60  # 16.4 min, initial 2 min will be ignored.
    # Here we set the fitparameter as GlobalCouple and excitatory recurrent strength
    GlobalCouple = fitparameter['GlobalCouple']
    ExcRec = fitparameter['ExcRec'] # excitatory recurrent strength
    ROI_num = np.size(SC,0)  # Number of ROIs, must matching the SC and FC.
    # Synaptic Parameters
    tau_s, r = 0.1, 0.641  # Unit: s, 1.
    # Connection Parameters
    J = 0.2609  # Unit: nA
    # F-I curve(activation function) Parameters
    a, b, d = 270.001, 108.001, 0.154  # Unit: n/C, Hz, ms, respectively.
    # Potential Fit Parameter
    w, I, sigma = ExcRec * np.ones(ROI_num), 0.3 * np.ones(ROI_num), 0.004 * np.ones(ROI_num)
    G = GlobalCouple
    # Initial conditions for synapse and current
    Synaptic = 0.02 * np.ones([ROI_num]) + np.random.normal(0, 0.01, size=ROI_num)  # labeled as S in paper
    Current = np.zeros([ROI_num])  # labeled as x in paper
    # Balloon-Windkessel hemodynamic model
    z, f, v, q = np.abs(np.random.normal(2, 0.5, size=ROI_num)), np.abs(np.random.normal(2, 1, size=ROI_num)), \
                 np.abs(np.random.normal(2, 0.5, size=ROI_num)), np.abs(np.random.normal(2, 1, size=ROI_num))
    Synaptic_Record = np.zeros((ROI_num, int(Simulation_During / dt)))
    for timepoint in range(0, int(Simulation_During / dt)):
        if 0 in (a * Current - b):
            x = a*Current-b
            x[np.where((a * Current - b) ==0)[0]] = 0.0001
            fr = x/(1-np.exp(-d*x))
        else:
            fr = (a * Current - b) / (1 - np.exp(-d * (a * Current - b)))  ## population firing rate
        dSynaptic = (-Synaptic / tau_s + r * (1 - Synaptic) * fr) * dt + \
                    (sigma * np.random.normal(0, 1, size=ROI_num)) * np.sqrt(dt)  # average synaptic gating variable
        Synaptic = Synaptic + dSynaptic
        Current = w * J * Synaptic + I + G * J * np.dot(SC, Synaptic)  ##total input current for i-th region
        Synaptic_Record[:, timepoint] = Synaptic
    return Synaptic_Record

def costfun(FC_real, FC_model, costfunmethod):
    # Input:
    #          FC_real: real FC matrix from fMRI data
    #          FC_model: simulated FC matrix from dynamic model
    #          costfunmethod: the method for the cost function. Currently, there are only FC correlation and FC distance methods.
    # Return:
    #         cost: the similarity between FC_real and FC_model, smaller cost, higher similarity
    ROI_num= np.size(FC_real, 0) # Number of ROIs
    if costfunmethod=='Corr':
        corr_model =np.corrcoef(np.reshape(FC_real, ROI_num ** 2), np.reshape(FC_model, ROI_num ** 2))[0][1]
        cost= 1-corr_model # find the min value of the cost function later.
    elif costfunmethod=='Distance':
        cost= np.sqrt(np.sum(np.power(np.reshape(FC_real, ROI_num ** 2)-np.reshape(FC_model, ROI_num ** 2),2)))/ ROI_num
    elif costfunmethod=='DFC':
        print('Currently, no such cost function right now.')
    return cost

def opt_CMA_ES(fitparameter_boundary,costfun_local,opt_maxstep=500):
    # opt by Evolution Strategy with Covariance Matrix Adaptation
    # INPUT
    # fitparameter_boundary: [[N,],[N,]] Low boundary and High boundary, respectively
    # costfun_local: should set the fitting parameters as input and return costfunction value
    parameter_num = len(fitparameter_boundary[0])
    x0 = np.random.rand(1,parameter_num)*(np.array(fitparameter_boundary[1])-np.array(fitparameter_boundary[0]))\
             + np.array(fitparameter_boundary[0]) # random choice the start point
    x_list_mean, x_list_all = [], []  # Record the mean/all parameter set in each step
    training_y_list_mean, training_y_list_all = [], []  # cost function of mean/all parameter set in each step
    es = cma.CMAEvolutionStrategy(x0, 0.2, {'maxiter': opt_maxstep, 'bounds': fitparameter_boundary}) # set CMA_ES
    i = 0
    while not es.stop():
        X = es.ask(); y = []
        for x in X:
            y.append(costfun_local(x))
        print('Here is step ' + str(i) + "\n")
        i = i + 1
        x_list_mean.append(np.array(X).mean(axis=0))
        x_list_all.append(np.array(X))
        training_y_list_mean.append(costfun_local(x_list_mean[-1]))
        training_y_list_all.append(np.array(y))
        es.tell(X, y)
    return {"x_list_mean":x_list_mean,"x_list_all":x_list_all,
            "training_y_list_mean":training_y_list_mean,"training_y_list_all":training_y_list_all}

def balloonWindkessel(z, sampling_rate,B0 = 3,r0 = 110,TE = 3.31):
    """
    Computes the Balloon-Windkessel transformed BOLD signal
    Numerical method (for integration): Runge-Kutta 2nd order method (RK2)
    z:          Measure of neuronal activity (space x time 2d array, or 1d time array)
    sampling_rate: sampling rate, or time step (in seconds)
    alpha:      Grubb's exponent
    kappa:      Rate of signal decay (in seconds)
    gamma:      Rate of flow-dependent estimation (in seconds)
    tau:        Hemodynamic transit time (in seconds)
    rho:        Resting oxygen extraction fraction
    V0:         resting blood vlume fraction
    RETURNS:
    BOLD:       The transformed BOLD signal (from neural/synaptic activity)
    s:          Vasodilatory signal
    f:          blood inflow
    v:          blood volume
    q:          deoxyhemoglobin content
    ## note the scanner parameters
    B0:         3T scanner
    r0:         intravascular relaxation rate as a function of oxygen saturation, about 110 Hz in a 3T MR scanner
    TE:
    """
    if z.ndim==2:
        timepoints = z.shape[1]
    else:
        timepoints = len(z)
        z.shape = (1,len(z))

    dt = sampling_rate
    # Constants
    # k1 = 7*rho
    # k2 = 2
    # k3 = 2*rho - 0.2
    v_0 = 28.265 * B0  ##frequency offset at the outer surface of magnetized vessels
    rho = 0.34  ##resting oxygen extraction fraction
    epsilon = 0.47  # the ratio between intravascular and extravascular MR signal
    k1 = 4.3 * v_0 * rho * TE  ##k1, k2, and k3  are parameters dependent on field strength
    k2 = epsilon * r0 * rho * TE
    k3 = 1 - epsilon
    alpha=0.32
    kappa=0.65
    gamma=0.41
    tau=0.98
    V0=0.02
    # Create lambda function to calculate E, flow
    E = lambda x: 1.0 - (1.0 - rho)**(1.0/x) # x is f, in this case
    # Create lambda function to calculate y, the BOLD signal
    y = lambda q1,v1: V0 * (k1*(1.0-q1) + k2*(1.0 - q1/v1) + k3*(1.0 - v1))

    # initialize empty matrices to integrate through
    BOLD = np.zeros(z.shape)
    s = np.zeros(z.shape) # vasodilatory signal
    f = np.zeros(z.shape) # blood inflow
    v = np.zeros(z.shape) # blood volume
    q = np.zeros(z.shape) # deoxyhemoglobin content

    # Set initial conditions
    s[:,0] = 0.0
    f[:,0] = 1.0
    v[:,0] = 1.0
    q[:,0] = 1.0
    BOLD[:,0] = y(q[:,0], v[:,0])

    ## Obtain mean value of z, and then calculate steady state of variables prior to performing HRF modeling
    z_mean = np.mean(z,axis=1)

    # Run loop until an approximate steady state is reached
    for t in range(timepoints-1):

        # 1st order increments (regular Euler)
        #s_k1 = z_mean - (1.0/kappa)*s[:,t] - (1.0/gamma)*(f[:,t] - 1.0)
        s_k1 = z_mean - (kappa)*s[:,t] - (gamma)*(f[:,t] - 1.0)
        f_k1 = s[:,t]
        v_k1 = (f[:,t] - v[:,t]**(1.0/alpha))/tau
        q_k1 = (f[:,t]*E(f[:,t])/rho - (v[:,t]**(1.0/alpha)) * q[:,t]/v[:,t])/tau

        # Compute intermediate values (Euler method)
        s_a = s[:,t] + s_k1*dt
        f_a = f[:,t] + f_k1*dt
        v_a = v[:,t] + v_k1*dt
        q_a = q[:,t] + q_k1*dt

        # 2nd order increments (RK2 method)
        #s_k2 = z_mean - (1.0/kappa)*s_a - (1.0/gamma)*(f_a - 1.0)
        s_k2 = z_mean - (kappa)*s_a - (gamma)*(f_a - 1.0)
        f_k2 = s_a
        v_k2 = (f_a - v_a**(1.0/alpha))/tau
        q_k2 = (f_a*E(f_a)/rho - (v_a**(1.0/alpha)) * q_a/v_a)/tau

        # Compute RK2 increment
        s[:,t+1] = s[:,t] + (.5*(s_k1+s_k2))*dt
        f[:,t+1] = f[:,t] + (.5*(f_k1+f_k2))*dt
        v[:,t+1] = v[:,t] + (.5*(v_k1+v_k2))*dt
        q[:,t+1] = q[:,t] + (.5*(q_k1+q_k2))*dt

        BOLD[:,t+1] = y(q[:,t+1], v[:,t+1])

        # If an approximate steady state is reached, quit.
        # We know HRF is at least 10 seconds, so make sure we wait at least 10 seconds until identifying a 'steady state'
        if (t*dt)>10 and np.sum(np.abs(BOLD[:,t+1]-BOLD[:,t]))==0: break

    ## After identifying steady state, re-initialize to run actual simulation
    s[:,0] = s[:,t+1]
    f[:,0] = f[:,t+1]
    v[:,0] = v[:,t+1]
    q[:,0] = q[:,t+1]
    BOLD[:,0] = y(q[:,t+1], v[:,t+1])

    for t in range(timepoints-1):

        # 1st order increments (regular Euler)
        #s_k1 = z[:,t] - (1.0/kappa)*s[:,t] - (1.0/gamma)*(f[:,t] - 1.0)
        s_k1 = z[:,t] - (kappa)*s[:,t] - (gamma)*(f[:,t] - 1.0)
        f_k1 = s[:,t]
        v_k1 = (f[:,t] - v[:,t]**(1.0/alpha))/tau
        q_k1 = (f[:,t]*E(f[:,t])/rho - (v[:,t]**(1.0/alpha)) * q[:,t]/v[:,t])/tau

        # Compute intermediate values (Euler method)
        s_a = s[:,t] + s_k1*dt
        f_a = f[:,t] + f_k1*dt
        v_a = v[:,t] + v_k1*dt
        q_a = q[:,t] + q_k1*dt

        # 2nd order increments (RK2 method)
        #s_k2 = z[:,t+1] - (1.0/kappa)*s_a - (1.0/gamma)*(f_a - 1.0)
        s_k2 = z[:,t+1] - (kappa)*s_a - (gamma)*(f_a - 1.0)
        f_k2 = s_a
        v_k2 = (f_a - v_a**(1.0/alpha))/tau
        q_k2 = (f_a*E(f_a)/rho - (v_a**(1.0/alpha)) * q_a/v_a)/tau

        # Compute RK2 increment
        s[:,t+1] = s[:,t] + (.5*(s_k1+s_k2))*dt
        f[:,t+1] = f[:,t] + (.5*(f_k1+f_k2))*dt
        v[:,t+1] = v[:,t] + (.5*(v_k1+v_k2))*dt
        q[:,t+1] = q[:,t] + (.5*(q_k1+q_k2))*dt

        BOLD[:,t+1] = y(q[:,t+1], v[:,t+1])

    return BOLD







