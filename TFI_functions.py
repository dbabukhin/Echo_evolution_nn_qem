import numpy as np


def X_evol(circuit, qr, h_list, T):
    """
    Implement Trotter step for Hx = h_{j}Sx_{j}
    
    h_list - list of local field values
    T==T_full/N - evolution time per step, N - Trotter steps number, T_full - full evolution time
    """
    
    for i, h in enumerate(h_list):
        if h != 0: circuit.rx(2*h*T, qr[i])
    circuit.barrier(qr)
    
    return circuit

def ZZ_evol(circuit, qr, coupling_map, J_lst, T):
    """
    Implement Trotter step for Hzz = J_{ij}Sz_{i}Sz_{j}
    
    J_list - list of coupling constants
    T==T_full/N - evolution time per step, N - Trotter steps number, T_full - full evolution time
    """
    
    circuit.barrier(qr)
    
    for c, J in zip(coupling_map[::-1], J_lst):
        circuit.barrier(qr)
        circuit.cx(qr[c[0]], qr[c[1]])
        circuit.rz(2*J*T, qr[c[1]])
        circuit.cx(qr[c[0]], qr[c[1]])
        circuit.barrier(qr)
        
    circuit.barrier(qr)
    return circuit
    
def Trotter_step(circuit, qr, coupling_map, h_list, J_list, T):
    """
    Implement Trotter step for Transverse field Ising hamiltonian
    
    h_list - list of local field values
    J_list - list of coupling constants
    T==T_full/N - evolution time per step, N - Trotter steps number, T_full - full evolution time
    """
    
    circuit = X_evol(circuit, qr, h_list, T)
    circuit = ZZ_evol(circuit, qr, coupling_map, J_list, T)
    
    circuit.barrier(qr)
    return circuit
    
def inverse_Trotter_step(circuit, qr, coupling_map, h_list, J_list, T):
    """
    Функция, реализующая один шаг Троттера в разложении оператора эволюции для гамильтониана модели Изинга
    
    h_list - лист значений локального поперечного поля
    J_list - лист значений констант парного взаимодействия спинов
    T==T_full/N - время эволюции, где N - число шагов Троттера, T_full - полное время эволюции
    """
    
    circuit = ZZ_evol(circuit, qr, coupling_map, J_list, -T)
    circuit = X_evol(circuit, qr, h_list, -T)
    
    circuit.barrier(qr)
    return circuit

def magnetization(counts_list):
    
    """
    Calculate average magnetization of the spin system
    """
    
    magn_lst = []
    for counts_dict in counts_list:
        m = 0.0
        for bits, counts in counts_dict.items():
            bit_count = 0
            for s in bits:
                bit_count += int(s) 
            m += bit_count*counts/8192
        magn_lst.append(m)
        
    return magn_lst
            
    
def spins_magnetization(counts_dict, num_qubits):
    
    """
    Calculate magnetization of single spins
    """    
    
    magnetization_array = np.zeros(num_qubits)
    full_counts = 0
    for bitstring, counts in counts_dict.items():
        full_counts += counts
    for bitstring, counts in counts_dict.items():
        for j, s in enumerate(bitstring):
             if s == "1":
                    magnetization_array[j] += counts/full_counts
        
    return magnetization_array