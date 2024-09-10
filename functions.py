import numpy as np
import pickle
import os
from datetime import date

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute
from qiskit.quantum_info import Statevector, Pauli
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import pauli_error, depolarizing_error, thermal_relaxation_error
from qiskit import Aer

import TFI_functions


def M_euclidean_distance(x, y):
    """
    Calculate euclidean distance of vectors x and y
    """

    distance = 0
    for x_i, y_i in zip(x,y):
        distance += (x_i - y_i)**2
    distance = np.sqrt(distance)
    distance /= len(x)
    return distance 


def M_average_distance(x, y):
    """
    Calculate difference of mean values of vectors x and y
    """

    x_aver = 0
    y_aver = 0
    for x_i, y_i in zip(x,y):
        x_aver += x_i
        y_aver += y_i
    x_aver /= len(x)
    y_aver /= len(y)
    return (x_aver - y_aver) 


def M_abs_average_distance(x, y):
    """
    Calculate absolute difference of mean values of vectors x and y
    """

    x_aver = 0
    y_aver = 0
    for x_i, y_i in zip(x,y):
        x_aver += x_i
        y_aver += y_i
    x_aver /= len(x)
    y_aver /= len(y)
    return np.abs(x_aver - y_aver) 


def generate_random_state(num_qubits, coupling_map, p_th=0.5):
    
    """
    Generate a random initial state. A layer on single-qubit u3 rotations
    with random parameters theta, phi and lambda is followed by a layer 
    of CNOT gates, applied with probability p_th to every couple of connected
    qubits.
    """

    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qr, cr)
    
    for i in range(num_qubits):
        theta   = np.arccos(np.random.uniform(-1, 1, 1)[0])
        phi     = np.random.uniform(0.0, 2*np.pi, 1)[0]
        lambd   = 0
        qc.u(theta, phi, lambd, i)
    
    for c in coupling_map:
        p = np.random.uniform(0.0, 1.0, 1)
        if p > p_th:
            qc.cnot(c[0], c[1])
        else:
            pass
        
    return qc


def generate_noise_model(noise_dict):
    
    """
    Generate a quantum noise model.
    """

    noise_model = NoiseModel()
    
    for noise_name, params in noise_dict.items():
        if noise_name == "depolarizing":
            error_1q = depolarizing_error(params[0], 1)
            error_2q = error_1q.tensor(error_1q)

            noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

        if noise_name == "pauli":
            p_x = params[1]
            p_y = params[2]
            p_z = params[3]
            p_id = params[0]
            assert p_x + p_y + p_z + p_id == 1.0, "Probability values do not sum to 1"
            
            error_1q = pauli_error( [ 
                (Pauli(z=[0], x=[0]), p_id),
                (Pauli(z=[0], x=[1]), p_x),
                (Pauli(z=[1], x=[1]), p_y),
                (Pauli(z=[1], x=[0]), p_z)
            ] )
            error_2q = error_1q.tensor(error_1q)

            noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
            
    return noise_model
    

def generate_data_TFI_echo_dynamics(
    sample_size=1,
    num_qubits=2, 
    num_trotters=1, 
    time=np.pi,
    time_points=1,
    h_array=[1.0, 1.0],
    J_array=[2.0],
    coupling_map=[[0,1]],
    p_th=0.5,
    noise_model=None,
    noise_dict=None,
    periodic_save=False):
    """
    Generate echo-evolution data (noisy and noise-free magnetization vectors) 
    for random initial states of Transverse-field Ising model. 
    """
    
    backend = Aer.get_backend('qasm_simulator')

    # Upload existing dataset to expand it with new data, else, generate a new one
    if periodic_save:
        now = date.today().isoformat()
        cwd = os.getcwd()
        new_dir = cwd + '\\' + now

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        else:
            pass

        noise_params = [v for v in noise_dict.values()]
        noise_params_ = []
        for p in noise_params:
            for par in p:
                noise_params_.append(par)

        data_name = (backend.name() + "_"
                    + "_".join(noise_dict.keys()) + "_" + "_".join(map(str, noise_params_)) 
                    + "_%d_qubits"%num_qubits + "_%d_trotters"%num_trotters
                    + "_%d_timesteps"%time_points)
        try:
            with open(new_dir + '\\' + data_name + '.pkl', 'rb') as f:
                data_loaded = pickle.load(f)
                print("existing data loaded")
        except:
            data_loaded = np.zeros((1, time_points, 2, num_qubits))
            print("no existing data loaded")
    data = np.zeros((sample_size, time_points, 2, num_qubits))

    time_array = np.linspace(1e-6, time, time_points)

    for j in range(sample_size):
    
        init_state = Statevector(generate_random_state(num_qubits=num_qubits, coupling_map=coupling_map, p_th=p_th))

        for k, t in enumerate(time_array):
            """
            Noise-free
            """
            qr1 = QuantumRegister(num_qubits)
            cr1 = ClassicalRegister(num_qubits)
            qc1 = QuantumCircuit(qr1, cr1)

            qc1.initialize(init_state, qr1)
            for _ in range(num_trotters):
                qc1 = TFI_functions.Trotter_step(qc1, qr1, coupling_map, h_array, J_array, t/(2*num_trotters))
            for _ in range(num_trotters):
                qc1 = TFI_functions.inverse_Trotter_step(qc1, qr1, coupling_map, h_array, J_array, t/(2*num_trotters))
            qc1.barrier(qr1)
            qc1.measure(qr1, cr1)

            job1 = execute(qc1, backend, shots=10000)
            result1 = job1.result().get_counts(qc1)

            data[j][k][0] = TFI_functions.spins_magnetization(result1, num_qubits)         

            """
            Noisy
            """
            qr2 = QuantumRegister(num_qubits)
            cr2 = ClassicalRegister(num_qubits)
            qc2 = QuantumCircuit(qr2, cr2)

            qc2.initialize(init_state, qr2)
            for _ in range(num_trotters):
                qc2 = TFI_functions.Trotter_step(qc2, qr2, coupling_map, h_array, J_array, t/(2*num_trotters))
            for _ in range(num_trotters):
                qc2 = TFI_functions.inverse_Trotter_step(qc2, qr2, coupling_map, h_array, J_array, t/(2*num_trotters))
            qc2.barrier(qr2)
            qc2.measure(qr2, cr2)

            if noise_model:
                job2 = execute(qc2, backend, shots=10000, basis_gates=noise_model.basis_gates, noise_model=noise_model)
            else:
                job2 = execute(qc2, backend, shots=10000)
            result2 = job2.result().get_counts(qc2)

            data[j][k][1] = TFI_functions.spins_magnetization(result2, num_qubits)     
        
        # Save generated dataset every 50 initial states
        if j%50 == 0:
            output = open(new_dir + '/{}.pkl'.format(data_name), 'wb')
            pickle.dump(np.vstack((data_loaded, data)), output)
            output.close()
            print(j)

    data_dict = {
        "data": np.vstack((data_loaded, data)),
        "parameters":
        {
            "init state": init_state,
            "num of qubits": num_qubits,
            "num of trotters": num_trotters,
            "total sim time": time,
            "time points": time_points,
            "h values": h_array,
            "J values": J_array,
            "coupling map": coupling_map,
            "p threshold": p_th,
            "circuit": qc1,
            "backend": job1.backend()         
        }
    }
    
    return data_dict


def generate_data_TFI_forward_dynamics(
    sample_size=1,
    num_qubits=2, 
    num_trotters=1,
    exact_multiplier=10,
    time=np.pi,
    time_points=1,
    h_array=[1.0, 1.0],
    J_array=[2.0],
    coupling_map=[[0,1]],
    p_th=0.5,
    noise_model=None):  
    """
    Generate forward-in-time-evolution data (noisy and noise-free magnetization vectors) 
    for random initial states of Transverse-field Ising model.
    """
    
    data = np.zeros((sample_size, time_points, 3, num_qubits))
    
    backend = Aer.get_backend('qasm_simulator')
    
    time_array = np.linspace(1e-6, time, time_points)
    
    for j in range(sample_size):
        
        init_state = Statevector(generate_random_state(num_qubits=num_qubits, coupling_map=coupling_map, p_th=p_th))
        
        for k, t in enumerate(time_array):
            """
            Noise-free
            """
            qr1 = QuantumRegister(num_qubits)
            cr1 = ClassicalRegister(num_qubits)
            qc1 = QuantumCircuit(qr1, cr1)

            qc1.initialize(init_state, qr1)
            for _ in range(num_trotters):
                qc1 = TFI_functions.Trotter_step(qc1, qr1, coupling_map, h_array, J_array, t/(2*num_trotters))
                qc1 = TFI_functions.Trotter_step(qc1, qr1, coupling_map, h_array, J_array, t/(2*num_trotters))
            qc1.barrier(qr1)
            qc1.measure(qr1, cr1)

            job1 = execute(qc1, backend, shots=10000)
            result1 = job1.result().get_counts(qc1)

            data[j][k][0] = TFI_functions.spins_magnetization(result1, num_qubits)         

            """
            Noisy
            """
            qr2 = QuantumRegister(num_qubits)
            cr2 = ClassicalRegister(num_qubits)
            qc2 = QuantumCircuit(qr2, cr2)

            qc2.initialize(init_state, qr2)
            for _ in range(num_trotters):
                qc2 = TFI_functions.Trotter_step(qc2, qr2, coupling_map, h_array, J_array, t/(2*num_trotters))
                qc2 = TFI_functions.Trotter_step(qc2, qr2, coupling_map, h_array, J_array, t/(2*num_trotters))
            qc2.barrier(qr2)
            qc2.measure(qr2, cr2)

            if noise_model:
                job2 = execute(qc2, backend, shots=10000, basis_gates=noise_model.basis_gates, noise_model=noise_model)
            else:
                job2 = execute(qc2, backend, shots=10000)
            result2 = job2.result().get_counts(qc2)

            data[j][k][1] = TFI_functions.spins_magnetization(result2, num_qubits)
            
            """
            Noise-free with increased number of Trotter steps (ground truth)
            """
            qr_gt = QuantumRegister(num_qubits)
            cr_gt = ClassicalRegister(num_qubits)
            qc_gt = QuantumCircuit(qr_gt, cr_gt)

            qc_gt.initialize(init_state, qr_gt)
            N_trotters_exact = int(exact_multiplier*num_trotters)
            for _ in range( N_trotters_exact ):
                qc_gt = TFI_functions.Trotter_step(qc_gt, qr_gt, coupling_map, h_array, J_array, t/(2*N_trotters_exact))
                qc_gt = TFI_functions.Trotter_step(qc_gt, qr_gt, coupling_map, h_array, J_array, t/(2*N_trotters_exact))
            qc_gt.barrier(qr_gt)
            qc_gt.measure(qr_gt, cr_gt)

            job_gt = execute(qc_gt, backend, shots=10000)
            result_gt = job_gt.result().get_counts(qc_gt)

            data[j][k][2] = TFI_functions.spins_magnetization(result_gt, num_qubits)    
        
        
    data_dict = {
        "data": data,
        "parameters":
        {
            "init state": init_state,
            "num of qubits": num_qubits,
            "num of trotters": num_trotters,
            "total sim time": time,
            "time points": time_points,
            "h values": h_array,
            "J values": J_array,
            "coupling map": coupling_map,
            "p threshold": p_th,
            "circuit": qc1,
            "backend": job1.backend()
        }
    }
    
    return data_dict


def test_generate_data_TFI_forward_dynamics(
    init_state=np.array([1.0, 0.0, 0.0, 0.0]),
    sample_size=1,
    num_qubits=2, 
    num_trotters=1,
    exact_multiplier=10,
    time=np.pi,
    time_points=1,
    h_array=[1.0, 1.0],
    J_array=[2.0],
    coupling_map=[[0,1]],
    p_th=0.5,
    noise_model=None):  
    """
    Generate forward-in-time-evolution data (noisy and noise-free magnetization vectors) 
    for random initial states of Transverse-field Ising model.
    """
    
    data = np.zeros((sample_size, time_points, 3, num_qubits))
    
    backend = Aer.get_backend('qasm_simulator')
    
    time_array = np.linspace(1e-6, time, time_points)
    
    for j in range(sample_size):
        
        #init_state = Statevector(generate_random_state(num_qubits=num_qubits, coupling_map=coupling_map, p_th=p_th))
        
        for k, t in enumerate(time_array):
            """
            Noise-free
            """
            qr1 = QuantumRegister(num_qubits)
            cr1 = ClassicalRegister(num_qubits)
            qc1 = QuantumCircuit(qr1, cr1)

            qc1.initialize(init_state, qr1)
            for _ in range(num_trotters):
                qc1 = TFI_functions.Trotter_step(qc1, qr1, coupling_map, h_array, J_array, t/(2*num_trotters))
                qc1 = TFI_functions.Trotter_step(qc1, qr1, coupling_map, h_array, J_array, t/(2*num_trotters))
            qc1.barrier(qr1)
            qc1.measure(qr1, cr1)

            job1 = execute(qc1, backend, shots=10000)
            result1 = job1.result().get_counts(qc1)

            data[j][k][0] = TFI_functions.spins_magnetization(result1, num_qubits)         

            """
            Noisy
            """
            qr2 = QuantumRegister(num_qubits)
            cr2 = ClassicalRegister(num_qubits)
            qc2 = QuantumCircuit(qr2, cr2)

            qc2.initialize(init_state, qr2)
            for _ in range(num_trotters):
                qc2 = TFI_functions.Trotter_step(qc2, qr2, coupling_map, h_array, J_array, t/(2*num_trotters))
                qc2 = TFI_functions.Trotter_step(qc2, qr2, coupling_map, h_array, J_array, t/(2*num_trotters))
            qc2.barrier(qr2)
            qc2.measure(qr2, cr2)

            if noise_model:
                job2 = execute(qc2, backend, shots=10000, basis_gates=noise_model.basis_gates, noise_model=noise_model)
            else:
                job2 = execute(qc2, backend, shots=10000)
            result2 = job2.result().get_counts(qc2)

            data[j][k][1] = TFI_functions.spins_magnetization(result2, num_qubits)
            
            """
            Noise-free with increased number of Trotter steps (ground truth)
            """
            qr_gt = QuantumRegister(num_qubits)
            cr_gt = ClassicalRegister(num_qubits)
            qc_gt = QuantumCircuit(qr_gt, cr_gt)

            qc_gt.initialize(init_state, qr_gt)
            N_trotters_exact = int(exact_multiplier*num_trotters)
            for _ in range( N_trotters_exact ):
                qc_gt = TFI_functions.Trotter_step(qc_gt, qr_gt, coupling_map, h_array, J_array, t/(2*N_trotters_exact))
                qc_gt = TFI_functions.Trotter_step(qc_gt, qr_gt, coupling_map, h_array, J_array, t/(2*N_trotters_exact))
            qc_gt.barrier(qr_gt)
            qc_gt.measure(qr_gt, cr_gt)

            job_gt = execute(qc_gt, backend, shots=10000)
            result_gt = job_gt.result().get_counts(qc_gt)

            data[j][k][2] = TFI_functions.spins_magnetization(result_gt, num_qubits)    
        
        
    data_dict = {
        "data": data,
        "parameters":
        {
            "init state": init_state,
            "num of qubits": num_qubits,
            "num of trotters": num_trotters,
            "total sim time": time,
            "time points": time_points,
            "h values": h_array,
            "J values": J_array,
            "coupling map": coupling_map,
            "p threshold": p_th,
            "circuit": qc1,
            "backend": job1.backend()
        }
    }
    
    return data_dict