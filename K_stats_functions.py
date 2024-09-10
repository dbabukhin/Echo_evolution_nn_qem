import numpy as np
import pickle 
import torch


def load_data(data_dir):
    
    file_name = 'trained_model_matrix'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    model_matrix_loaded = pickle.load(file)
    file.close()

    file_name = 'train_losses'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    train_losses_array = pickle.load(file)
    file.close()

    file_name = 'valid_losses'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    valid_losses_array = pickle.load(file)
    file.close()

    file_name = 'train_data_samples'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    train_data_samples = pickle.load(file)
    file.close()

    file_name = 'valid_data_samples'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    valid_data_samples = pickle.load(file)
    file.close()

    file_name = 'test_data_samples'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    test_data_samples = pickle.load(file)
    file.close()

    file_name = 'forward_data_samples'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    forward_data_samples = pickle.load(file)
    file.close()
    
    file_name = 'time_points'
    file = open(data_dir + '/{}.pkl'.format(file_name), 'rb')
    time_points = pickle.load(file)
    file.close()
    
    res_dict = {
    "train":        train_data_samples,
    "valid":        valid_data_samples,
    "test":         test_data_samples,
    "forward":      forward_data_samples,
    "time":         time_points,
    "models":       model_matrix_loaded,
    "train losses": train_losses_array,
    "valid losses": valid_losses_array
    }
    
    return res_dict


def calculate_M_K_values_echo(model_matrix, X_test_samples_array, y_test_samples_array, eps=1e-2):
    
    N_samples = X_test_samples_array.shape[0] # Number of data samples
    N_states  = X_test_samples_array.shape[1] # Number of initial states in a sample
    N_Dhidden = len(model_matrix)             # Number of hidden layer widths 
    
    K_echo_array        = np.zeros((N_Dhidden, N_samples, N_states))
    abs_delta_M_before  = np.zeros((N_Dhidden, N_samples, N_states))
    abs_delta_M_after   = np.zeros((N_Dhidden, N_samples, N_states))
    
    eps = 1e-2
    
    for k, models in enumerate(model_matrix):
        for l, (model, X_test, y_test) in enumerate(zip(models, X_test_samples_array, y_test_samples_array)):

            # Samples of 1000 magnetization vectors of 6 spins
            X_test = torch.from_numpy(X_test)
            y_test = torch.from_numpy(y_test)

            # Average system magnetization arrays
            M_ideal     = np.zeros(N_states)
            M_noisy     = np.zeros(N_states)
            M_corrected = np.zeros(N_states)

            with torch.no_grad():
                for i, (x, y) in enumerate(zip(X_test, y_test)):
                    # x, y, y_ - magnetization vectors
                    y_ = model(x)

                    M_ideal[i]     = np.mean(y.numpy())
                    M_noisy[i]     = np.mean(x.numpy())
                    M_corrected[i] = np.mean(y_.numpy())

            abs_delta_M_before[k][l] = np.abs(M_ideal - M_noisy)
            abs_delta_M_after[k][l]  = np.abs(M_ideal - M_corrected)
            K_echo_array[k][l]       = 1 - abs_delta_M_after[k][l]/(abs_delta_M_before[k][l]+eps)
            
    """
    Calculation of %K>0
    
    K_echo_array.shape = (N_Dhidden, N_samples, N_states)
    """
    K_positive = np.zeros(K_echo_array.shape[0:2])
    for i in range(K_positive.shape[0]):
        for j in range(K_positive.shape[1]):
            # Calculate how many states in each sample were corrected with positive
            # error correction efficiency K (i.e., were not worsened by a neural network)
            K_positive[i][j] = len(K_echo_array[i][j][K_echo_array[i][j] > 0])/len(K_echo_array[i][j])

    # Calculate average number of positive correction efficiency over samples
    K_positive_mean = np.mean(K_positive, axis=1) 
    K_positive_std  = np.std(K_positive, axis=1, ddof=1)
    
    """
    Statistics for |delta M|
    
    abs_delta_M_before.shape = (N_Dhidden, N_samples, N_states)
    abs_delta_M_after.shape  = (N_Dhidden, N_samples, N_states)
    """
    # before
    abs_delta_M_before_mean = np.mean(abs_delta_M_before, axis=2)
    abs_delta_M_before_std  = np.std(abs_delta_M_before, axis=2, ddof=1)
    
    abs_delta_M_before_max    = np.max(abs_delta_M_before, axis=2)
    abs_delta_M_before_median = np.median(abs_delta_M_before, axis=2)
    abs_delta_M_before_min    = np.min(abs_delta_M_before, axis=2)
    
    # after
    abs_delta_M_after_mean  = np.mean(abs_delta_M_after, axis=2)
    abs_delta_M_after_std   = np.std(abs_delta_M_after, axis=2, ddof=1)

    abs_delta_M_after_max    = np.max(abs_delta_M_after, axis=2)
    abs_delta_M_after_median = np.median(abs_delta_M_after, axis=2)
    abs_delta_M_after_min    = np.min(abs_delta_M_after, axis=2)
    

    res_dict = {
        "%K>0": {"mean": K_positive_mean, "std": K_positive_std},
        "delta M before": {
            "mean":   abs_delta_M_before_mean, 
            "std":    abs_delta_M_before_std, 
            "max":    abs_delta_M_before_max, 
            "median": abs_delta_M_before_median, 
            "min":    abs_delta_M_before_min
        },
        "delta M after": {
            "mean":   abs_delta_M_after_mean, 
            "std":    abs_delta_M_after_std, 
            "max":    abs_delta_M_after_max, 
            "median": abs_delta_M_after_median, 
            "min":    abs_delta_M_after_min
        }
    }
    
    return res_dict