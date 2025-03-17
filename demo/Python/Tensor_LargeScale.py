import numpy as np
import scipy
import cma
import tensorflow_probability as tfp
import tensorflow as tf

def Stuart_Landau_model(regional, simula_time, dt, A, SC, W, G, beta):
    """
    Stuart-Landau model implementation in TensorFlow
    Corrected for element-wise multiplication and random seed handling
    """
    SC = tf.convert_to_tensor(SC, dtype=tf.float32)
    W = tf.convert_to_tensor(W, dtype=tf.float32)
    
    t_0 = 0
    t_steps = int((simula_time - t_0) / dt + 1)
    dsig = tf.sqrt(dt) * beta
    
    X = tf.TensorArray(tf.float32, size=t_steps)
    Y = tf.TensorArray(tf.float32, size=t_steps)

    X = X.write(0, tf.zeros(regional, dtype=tf.float32))
    Y = Y.write(0, tf.zeros(regional, dtype=tf.float32))

    def body(t, X, Y):
        X_prev = X.read(t-1)
        Y_prev = Y.read(t-1)
        
        Diff_X = tf.tile([X_prev], [regional, 1]) - tf.transpose(tf.tile([X_prev], [regional, 1]))
        
        # Use t as seed for X, matching NumPy implementation
        noise1 = tf.random.stateless_normal(
            shape=(regional,), 
            seed=[t, 0],  # Using t as part of the seed
            dtype=tf.float32
        )
        
        # CORRECTED: Element-wise multiplication instead of matrix-vector
        dX = dt * ((A - tf.pow(X_prev,2) - tf.pow(Y_prev,2)) * X_prev - 
                 W * Y_prev + 
                 G * tf.reduce_sum(SC * Diff_X, axis=1)) + dsig * noise1
        
        Diff_Y = tf.tile([Y_prev], [regional, 1]) - tf.transpose(tf.tile([Y_prev], [regional, 1]))
        
        # Use t-1 as seed for Y, matching NumPy implementation
        noise2 = tf.random.stateless_normal(
            shape=(regional,), 
            seed=[t-1, 0],  # Using t-1 as part of the seed
            dtype=tf.float32
        )
        
        # CORRECTED: Element-wise multiplication instead of matrix-vector
        dY = dt * ((A - tf.pow(X_prev,2) - tf.pow(Y_prev,2)) * Y_prev + 
                 W * X_prev + 
                 G * tf.reduce_sum(SC * Diff_Y, axis=1)) + dsig * noise2
        
        X = X.write(t, X_prev + dX)
        Y = Y.write(t, Y_prev + dY)
        return t+1, X, Y

    t = tf.constant(1)
    _, X_final, Y_final = tf.while_loop(
        cond=lambda t, *_: t < t_steps,
        body=body,
        loop_vars=[t, X, Y],
        maximum_iterations=t_steps-1
    )
    
    Signal = X_final.stack()
    return Signal

def Linear_gaussian_model(fitparameter, SC):
    beta = fitparameter['beta']
    G = fitparameter['G']
    a = fitparameter['a']
    N_ROI = 360
    
    SC = tf.pow(SC, beta)
    H = tf.linalg.diag(tf.reduce_sum(SC, axis=1)) - SC
    SE, _ = tf.linalg.eig(H)
    H = H / tf.reduce_max(tf.math.real(SE))
    
    Q = tf.sqrt(2.0) * tf.linalg.inv(tf.eye(N_ROI, dtype=tf.float32)*a + G*H)
    Cov = Q @ tf.transpose(Q)
    D = tf.linalg.diag(tf.sqrt(tf.linalg.diag_part(Cov)))
    D_inv = tf.linalg.inv(D)
    FC_model = D_inv @ Cov @ D_inv
    return FC_model

def NMDA_neural_mess_model(fitparameter, SC):
    """
    NMDA neural mass model implementation in TensorFlow
    Corrected for random number seeding
    """
    dt = 0.01
    Simulation_During = 16.4 * 60
    GlobalCouple = fitparameter['GlobalCouple']
    ExcRec = fitparameter['ExcRec']
    ROI_num = tf.shape(SC)[0]

    tau_s, r = 0.1, 0.641
    J = 0.2609
    a, b, d = 270.001, 108.001, 0.154
    w = ExcRec * tf.ones(ROI_num, dtype=tf.float32)
    I, sigma = 0.3 * tf.ones(ROI_num), 0.004 * tf.ones(ROI_num)
    G = GlobalCouple

    # Initialize with same random pattern as NumPy version
    init_noise = tf.random.normal(shape=(ROI_num,), stddev=0.01)
    Synaptic = 0.02 * tf.ones(ROI_num) + init_noise
    Current = tf.zeros(ROI_num)
    
    steps = int(Simulation_During / dt)
    Synaptic_Record = tf.TensorArray(tf.float32, size=steps)
    
    def body(timepoint, Synaptic, Current, Synaptic_Record):
        x = a * Current - b
        fr = tf.where(
            tf.equal(x, 0),
            0.0001,
            x / (1 - tf.exp(-d * x))
        )
        
        # Use timepoint as part of the seed for noise generation
        noise = tf.random.stateless_normal(
            shape=(ROI_num,), 
            seed=[timepoint, 0],
            dtype=tf.float32
        )
        
        dSynaptic = (-Synaptic / tau_s + r * (1 - Synaptic) * fr) * dt + \
                   sigma * noise * tf.sqrt(dt)
        Synaptic_new = Synaptic + dSynaptic
        Current_new = w * J * Synaptic_new + I + G * J * tf.linalg.matvec(SC, Synaptic_new)
        
        Synaptic_Record = Synaptic_Record.write(timepoint, Synaptic_new)
        return timepoint+1, Synaptic_new, Current_new, Synaptic_Record

    _, _, _, Synaptic_Record = tf.while_loop(
        cond=lambda t, *_: t < steps,
        body=body,
        loop_vars=[0, Synaptic, Current, Synaptic_Record],
        maximum_iterations=steps
    )
    
    return Synaptic_Record.stack()

def create_bounded_optimizer(initial_params, param_bounds, learning_rate=0.1):

    min_val = tf.constant([b[0] for b in param_bounds], dtype=tf.float32)
    max_val = tf.constant([b[1] for b in param_bounds], dtype=tf.float32)
    
    # 初始参数归一化到[0,1]
    normalized_init = (initial_params - min_val) / (max_val - min_val)
    
    # 使用反向sigmoid映射到无约束空间
    logit_init = tf.math.log(normalized_init / (1 - normalized_init))
    
    # 创建可训练变量
    logit_params = tf.Variable(logit_init, dtype=tf.float32)
    
    # 参数转换函数
    def get_params():
        normalized = tf.sigmoid(logit_params)
        return min_val + normalized * (max_val - min_val)
    
    return get_params, logit_params, tf.optimizers.Adam(learning_rate)

@tf.function
def EI_dMFM(SC, dt, TR, w, I_ext, G, sigma, H_e, H_i, tau):
    """
    Simplified implementation of the Excitatory-Inhibitory Dynamic Mean Field Model
    that returns the excitatory input directly instead of recording to a TensorArray.
    """
    n = tf.shape(SC)[0]
    w_e, w_i, w_ee, w_ei, w_ie = tf.unstack(w, axis=1)
    tau_e, tau_i = tf.unstack(tau, axis=1)
    
    # Initialize states
    S_e = tf.random.uniform(shape=[n], minval=0.01, maxval=0.1)
    S_i = tf.random.uniform(shape=[n], minval=0.01, maxval=0.1)
    
    # Use fixed seed for reproducibility
    seed = tf.constant([42, 0], dtype=tf.int32)
    
    # Simulation steps
    steps = tf.cast(TR/dt, tf.int32)
    
    # Run simulation for one TR period
    for i in range(steps):
        # Calculate excitatory and inhibitory inputs
        I_E = w_e*G*tf.linalg.matvec(SC, S_e) + w_ee*S_e - w_ie*S_i + I_ext
        I_I = w_i*G*tf.linalg.matvec(SC, S_e) + w_ei*S_e - w_i*S_i + I_ext
        
        # Apply activation functions
        r_e = H_e(I_E)
        r_i = H_i(I_I)
        
        # Generate noise with incrementing seed
        new_seed = tf.add(seed, i)
        noise_e = tf.random.stateless_normal(shape=[n], seed=new_seed, stddev=1.0)
        noise_i = tf.random.stateless_normal(shape=[n], seed=tf.add(new_seed, 1000), stddev=1.0)
        
        # Update synaptic states
        dS_e = (-S_e/tau_e + (1 - S_e)*r_e)*dt + sigma*noise_e*tf.sqrt(dt)
        dS_i = (-S_i/tau_i + (1 - S_i)*r_i)*dt + sigma*noise_i*tf.sqrt(dt)
        
        S_e = tf.clip_by_value(S_e + dS_e, 0.0, 1.0)
        S_i = tf.clip_by_value(S_i + dS_i, 0.0, 1.0)
    
    # Return the final excitatory input after simulation
    return w_e*G*tf.linalg.matvec(SC, S_e) + w_ee*S_e - w_ie*S_i + I_ext

def tf_cost_function(params, model_func, SC, FC_real, costfunmethod):
    fit_params = {
        'beta': params[0],
        'G': params[1],
        'a': params[2],
        'GlobalCouple': params[3],
        'ExcRec': params[4]
    }
    
    FC_model = model_func(fit_params, SC)

    return costfun(FC_real, FC_model, costfunmethod)

def opt_TensorFlow(model_func, fitparameter_boundary, SC, FC_real, 
                  costfunmethod='Corr', max_iter=500, learning_rate=0.1):
    
    param_bounds = np.array(fitparameter_boundary).T
    initial_params = np.random.rand(len(param_bounds)) * (param_bounds[:,1]-param_bounds[:,0]) + param_bounds[:,0]
    
    get_params, logit_params, optimizer = create_bounded_optimizer(
        initial_params, param_bounds, learning_rate
    )
    
    SC_tf = tf.convert_to_tensor(SC, dtype=tf.float32)
    FC_real_tf = tf.convert_to_tensor(FC_real, dtype=tf.float32)
    
    history = {
        'params': [],
        'costs': [],
        'grad_norms': []
    }

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            current_params = get_params()
            cost = tf_cost_function(current_params, model_func, SC_tf, FC_real_tf, costfunmethod)
        grads = tape.gradient(cost, [logit_params])
        optimizer.apply_gradients(zip(grads, [logit_params]))
        return cost, grads[0]
    for step in range(max_iter):
        cost, grad = train_step()
        current_params = get_params().numpy()
        history['params'].append(current_params)
        history['costs'].append(cost.numpy())
        history['grad_norms'].append(tf.norm(grad).numpy())
        
        if step % 50 == 0:
            print(f"Step {step}: Cost={cost.numpy():.4f}, Grad Norm={tf.norm(grad):.4f}")
    
    return {
        'optimal_params': current_params,
        'history': history,
        'final_cost': cost.numpy()
    }

def Balloon_Windkessel_model(z, sampling_rate, B0=3.0, r0=110.0, TE=3.31):
    """
    Balloon-Windkessel model for hemodynamic transformation
    Function renamed to match import in Test_Tensor.py
    """
    original_shape = z.shape
    if z.shape.ndims == 1:
        z = tf.expand_dims(z, 0)
    ROI_num, timepoints = tf.unstack(tf.shape(z))

    dt = tf.cast(sampling_rate, tf.float32)
    B0 = tf.cast(B0, tf.float32)
    r0 = tf.cast(r0, tf.float32)
    TE = tf.cast(TE, tf.float32)

    v_0 = 28.265 * B0
    rho = tf.constant(0.34, dtype=tf.float32)
    epsilon = tf.constant(0.47, dtype=tf.float32)
    alpha = tf.constant(0.32, dtype=tf.float32)
    kappa = tf.constant(0.65, dtype=tf.float32)
    gamma = tf.constant(0.41, dtype=tf.float32)
    tau = tf.constant(0.98, dtype=tf.float32)
    V0 = tf.constant(0.02, dtype=tf.float32)

    k1 = 4.3 * v_0 * rho * TE
    k2 = epsilon * r0 * rho * TE
    k3 = 1.0 - epsilon

    def initialize_states():
        return (
            tf.zeros((ROI_num,), dtype=tf.float32),  # s
            tf.ones((ROI_num,), dtype=tf.float32),   # f
            tf.ones((ROI_num,), dtype=tf.float32),   # v
            tf.ones((ROI_num,), dtype=tf.float32),   # q
            tf.zeros((ROI_num,), dtype=tf.float32)   # BOLD
        )

    BOLD_ta = tf.TensorArray(tf.float32, size=timepoints)
    states_ta = tf.TensorArray(tf.float32, size=timepoints, element_shape=(4, ROI_num))

    def E(f):
        return 1.0 - tf.pow(1.0 - rho, 1.0/f)

    def compute_BOLD(q, v):
        return V0 * (k1*(1.0-q) + k2*(1.0 - q/v) + k3*(1.0 - v))

    def steady_state_loop(i, s, f, v, q):
        z_mean = tf.reduce_mean(z, axis=1)
        
        s_k1 = z_mean - kappa*s - gamma*(f - 1.0)
        f_k1 = s
        v_k1 = (f - tf.pow(v, 1.0/alpha)) / tau
        q_k1 = (f*E(f)/rho - tf.pow(v, 1.0/alpha)*q/v) / tau

        s_a = s + s_k1*dt
        f_a = f + f_k1*dt
        v_a = v + v_k1*dt
        q_a = q + q_k1*dt

        s_k2 = z_mean - kappa*s_a - gamma*(f_a - 1.0)
        f_k2 = s_a
        v_k2 = (f_a - tf.pow(v_a, 1.0/alpha)) / tau
        q_k2 = (f_a*E(f_a)/rho - tf.pow(v_a, 1.0/alpha)*q_a/v_a) / tau

        s_new = s + 0.5*(s_k1 + s_k2)*dt
        f_new = f + 0.5*(f_k1 + f_k2)*dt
        v_new = v + 0.5*(v_k1 + v_k2)*dt
        q_new = q + 0.5*(q_k1 + q_k2)*dt
        
        BOLD_new = compute_BOLD(q_new, v_new)
        
        # Added check for exact equality to match NumPy behavior
        exact_match = tf.reduce_all(tf.equal(BOLD_new, compute_BOLD(q, v)))
        cond = tf.logical_and(
            tf.cast(i, tf.float32)*dt > 10.0,
            tf.logical_or(
                exact_match,
                tf.reduce_all(tf.abs(BOLD_new - compute_BOLD(q, v)) < 1e-6)
            )
        )
        
        return tf.cond(cond, 
                      lambda: (i, s_new, f_new, v_new, q_new),
                      lambda: (i+1, s_new, f_new, v_new, q_new))

    _, s_ss, f_ss, v_ss, q_ss = tf.while_loop(
        cond=lambda i, *_: i < 10000, 
        body=steady_state_loop,
        loop_vars=(0, *initialize_states()[:4]),
        maximum_iterations=10000
    )

    def main_loop(t, s, f, v, q, BOLD_ta, states_ta):
        z_t = tf.gather(z, t, axis=1)
        
        s_k1 = z_t - kappa*s - gamma*(f - 1.0)
        f_k1 = s
        v_k1 = (f - tf.pow(v, 1.0/alpha)) / tau
        q_k1 = (f*E(f)/rho - tf.pow(v, 1.0/alpha)*q/v) / tau

        s_a = s + s_k1*dt
        f_a = f + f_k1*dt
        v_a = v + v_k1*dt
        q_a = q + q_k1*dt

        s_k2 = tf.gather(z, t+1, axis=1) - kappa*s_a - gamma*(f_a - 1.0)
        f_k2 = s_a
        v_k2 = (f_a - tf.pow(v_a, 1.0/alpha)) / tau
        q_k2 = (f_a*E(f_a)/rho - tf.pow(v_a, 1.0/alpha)*q_a/v_a) / tau

        s_new = s + 0.5*(s_k1 + s_k2)*dt
        f_new = f + 0.5*(f_k1 + f_k2)*dt
        v_new = v + 0.5*(v_k1 + v_k2)*dt
        q_new = q + 0.5*(q_k1 + q_k2)*dt
        
        BOLD_new = compute_BOLD(q_new, v_new)

        BOLD_ta = BOLD_ta.write(t, BOLD_new)
        states_ta = states_ta.write(t, tf.stack([s_new, f_new, v_new, q_new]))
        
        return t+1, s_new, f_new, v_new, q_new, BOLD_ta, states_ta

    final_t, _, _, _, _, BOLD_ta, states_ta = tf.while_loop(
        cond=lambda t, *_: t < timepoints-1,
        body=main_loop,
        loop_vars=(
            0, 
            s_ss, f_ss, v_ss, q_ss,
            BOLD_ta.write(0, compute_BOLD(q_ss, v_ss)),
            states_ta.write(0, tf.stack([s_ss, f_ss, v_ss, q_ss]))
        ),
        maximum_iterations=timepoints-1
    )

    BOLD = BOLD_ta.stack()
    
    if original_shape.ndims == 1:
        BOLD = tf.squeeze(BOLD, axis=0)
    
    return BOLD
