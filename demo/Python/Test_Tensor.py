import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Tensor_LargeScale import Balloon_Windkessel_model

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("No GPU found, using CPU instead")

os.makedirs("output", exist_ok=True)
os.makedirs("Figures", exist_ok=True)

# Simplified EI_dMFM function
def EI_dMFM(SC, dt, TR, w, I_ext, G, sigma, H_e, H_i, tau):
    n = tf.shape(SC)[0]
    w_e, w_i, w_ee, w_ei, w_ie = tf.unstack(w, axis=1)
    tau_e, tau_i = tf.unstack(tau, axis=1)
    
    # Initialize states
    S_e = tf.random.uniform(shape=[n], minval=0.01, maxval=0.1)
    S_i = tf.random.uniform(shape=[n], minval=0.01, maxval=0.1)
    
    seed = tf.constant([42, 0], dtype=tf.int32)
    steps = tf.cast(TR/dt, tf.int32)
    
    for i in range(steps):
        I_E = w_e*G*tf.linalg.matvec(SC, S_e) + w_ee*S_e - w_ie*S_i + I_ext
        I_I = w_i*G*tf.linalg.matvec(SC, S_e) + w_ei*S_e - w_i*S_i + I_ext
        
        r_e = H_e(I_E)
        r_i = H_i(I_I)
        
        new_seed = tf.add(seed, i)
        noise_e = tf.random.stateless_normal(shape=[n], seed=new_seed, stddev=1.0)
        noise_i = tf.random.stateless_normal(shape=[n], seed=tf.add(new_seed, 1000), stddev=1.0)
        
        dS_e = (-S_e/tau_e + (1 - S_e)*r_e)*dt + sigma*noise_e*tf.sqrt(dt)
        dS_i = (-S_i/tau_i + (1 - S_i)*r_i)*dt + sigma*noise_i*tf.sqrt(dt)
        
        S_e = tf.clip_by_value(S_e + dS_e, 0.0, 1.0)
        S_i = tf.clip_by_value(S_i + dS_i, 0.0, 1.0)
    
    # Return excitatory input and reshape for Balloon model
    I_E = w_e*G*tf.linalg.matvec(SC, S_e) + w_ee*S_e - w_ie*S_i + I_ext
    return I_E

def H_e(x):
    return tf.where(tf.equal(x, 0), tf.constant(0.0, dtype=tf.float32),
                    x / (1 - tf.exp(-0.16 * (x - 125.0))))

def H_i(x):
    return tf.where(tf.equal(x, 0), tf.constant(0.0, dtype=tf.float32),
                    x / (1 - tf.exp(-0.087 * (x - 177.0))))

def load_and_preprocess_data(sc_path, fc_path):
    SC = np.loadtxt(sc_path, delimiter=',')
    FC = np.loadtxt(fc_path, delimiter=',')
    SC = SC / np.max(SC)
    n = SC.shape[0]
    mask = np.triu_indices(n, 1)
    FC[np.eye(n) == 1] = 0
    FC[FC < 0] = 0
    FC_vector = FC[mask]
    return SC, FC, FC_vector, n, mask

def parameter_search():
    start_time = time.time()
    
    # Load data
    SC, FC, FC_vector, n, mask = load_and_preprocess_data(
        "input/Desikan_input/sc_train.csv",
        "input/Desikan_input/fc_train.csv"
    )

    # Define parameter ranges
    w_ie_range = np.arange(0.5, 3.05, 0.05)
    G_range = np.arange(0.5, 3.05, 0.05)
    
    W_IE, G = np.meshgrid(w_ie_range, G_range)
    w_ie_flat = W_IE.flatten()
    g_flat = G.flatten()
    num_params = len(w_ie_flat)
    
    cc_bold = np.zeros(num_params)
    
    # Simulation parameters
    dt = 0.001
    T = 660
    TR = 0.7
    pre_time = 60
    
    observe_steps = int(np.ceil((T - pre_time) / TR))
    pre_steps = int(np.ceil(pre_time / TR))
    
    sigma = 0.01
    I_ext = 0.0
    
    # Process smaller batches to conserve memory
    batch_size = 1  # Reduced batch size due to memory constraints

    for batch_start in range(0, num_params, batch_size):
        batch_end = min(batch_start + batch_size, num_params)
        current_batch_size = batch_end - batch_start
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(num_params+batch_size-1)//batch_size} "
              f"(parameters {batch_start+1}-{batch_end})")
        
        # Process one parameter at a time
        for param_idx in range(batch_start, batch_end):
            i = param_idx - batch_start  # Local index in batch
            
            # Set up parameters for current model
            w_batch = np.ones((n, 5), dtype=np.float32)
            tau_batch = np.zeros((n, 2), dtype=np.float32)
            
            w_batch[:, 0] = 1.0  # w_e
            w_batch[:, 1] = 0.7  # w_i
            w_batch[:, 2] = 1.4  # w_ee
            w_batch[:, 3] = 1.0  # w_ei
            w_batch[:, 4] = w_ie_flat[param_idx]  # w_ie
            
            tau_batch[:, 0] = 0.1  # tau_e
            tau_batch[:, 1] = 0.01  # tau_i
            
            w_tf = tf.convert_to_tensor(w_batch, dtype=tf.float32)
            tau_tf = tf.convert_to_tensor(tau_batch, dtype=tf.float32)
            
            # Prepare inputs for current model
            SC_tf = tf.convert_to_tensor(SC, dtype=tf.float32)
            G_tf = tf.ones(n, dtype=tf.float32) * g_flat[param_idx]
            
            # Pre-simulation (warm-up)
            print(f"  Parameter {param_idx+1}/{num_params}: w_ie={w_ie_flat[param_idx]:.2f}, G={g_flat[param_idx]:.2f}")
            print("  Starting pre-simulation...")
            
            # Initialize neural activity
            activity = np.zeros((n, observe_steps))
            
            # Run pre-simulation
            for step in range(pre_steps):
                if step % 10 == 0:
                    print(f"    Pre-step {step+1}/{pre_steps}")
                I_E = EI_dMFM(SC_tf, dt, TR, w_tf, I_ext, G_tf, sigma, H_e, H_i, tau_tf)
            
            # Main simulation with recording
            print("  Starting main simulation...")
            for step in range(observe_steps):
                if step % 10 == 0:
                    print(f"    Step {step+1}/{observe_steps}")
                I_E = EI_dMFM(SC_tf, dt, TR, w_tf, I_ext, G_tf, sigma, H_e, H_i, tau_tf)
                
                # Add time dimension
                I_E_reshaped = tf.expand_dims(I_E, axis=1)  # Shape: [n, 1]
                
                # Get BOLD signal and store
                bold = Balloon_Windkessel_model(I_E_reshaped, dt)
                activity[:, step] = bold.numpy()
            
            # Calculate FC from activity
            FC_sim = np.corrcoef(activity)
            FC_sim[np.eye(n) == 1] = 0
            FC_sim[FC_sim < 0] = 0
            FC_sim_vector = FC_sim[mask]
            
            # Calculate correlation with empirical FC
            cc = np.corrcoef(FC_vector, FC_sim_vector)[0, 1]
            cc_bold[param_idx] = cc
            
            print(f"  Result: cc={cc:.4f}")
    
    # Reshape and plot results
    cc_bold_map = np.zeros((len(G_range), len(w_ie_range)))
    for i in range(num_params):
        w_ie_idx = np.where(w_ie_range == w_ie_flat[i])[0][0]
        g_idx = np.where(G_range == g_flat[i])[0][0]
        cc_bold_map[g_idx, w_ie_idx] = cc_bold[i]
    
    # Plot results
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cc_bold_map, cmap='turbo', 
                    xticklabels=np.round(w_ie_range, 2),
                    yticklabels=np.round(G_range, 2))
    ax.set_xlabel('w_ie')
    ax.set_ylabel('G')
    ax.set_title('Correlation between simulated and empirical FC')
    plt.tight_layout()
    plt.savefig("Figures/EI_dMFM_2d_para_search_TF.png")
    
    # Save results
    np.save("output/EI_dMFM_2d_para_search_TF.npy", cc_bold_map)
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    
    return cc_bold_map, w_ie_range, G_range, total_time

if __name__ == "__main__":
    parameter_search()