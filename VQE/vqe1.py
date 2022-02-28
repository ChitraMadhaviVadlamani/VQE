import numpy as np
from random import random
from scipy.optimize import minimize

#print(bottom^top,"@@") tensor product

from qiskit import *
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver

def hamiltonian_operator(a, b, c, d):
    """
    Creates a*I + b*Z + c*X + d*Y pauli sum 
    that will be our Hamiltonian operator.
    
    """
    #since no i the imaginary coefficients are set to 0
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": a}, "label": "I"},  #a is the weight and I is the Pauli Operator
                   {"coeff": {"imag": 0.0, "real": b}, "label": "Z"},
                   {"coeff": {"imag": 0.0, "real": c}, "label": "X"},
                   {"coeff": {"imag": 0.0, "real": d}, "label": "Y"}
                   ]
    }
    #print(pauli_dict,"Paulis Dict")
    return WeightedPauliOperator.from_dict(pauli_dict)
    #return WeightedPauliOperator.from_dict(pauli_dict,"Pauli_dict 1")


scale = 10
a, b, c, d = (scale*random(), scale*random(), 
              scale*random(), scale*random())
#print(a," ", b, " ", c, " ",d)
H = hamiltonian_operator(a, b, c, d) #method to make hamiltonians divided
#print("Here",H," This is the Hamiltonian 2",H.paulis)



#classical algorithm to find lowest energy state
exact_result = NumPyEigensolver(H).run()
#print(exact_result,"Exact result 3")
reference_energy = min(np.real(exact_result.eigenvalues))
#reference_state = min(np.real(exact_result.eigenstates))
print('The exact ground state energy is: {}'.format(reference_energy))
#print('The exact ground state  is: {}'.format(reference_state))


#circuit preparation for anstaz
def quantum_state_preparation(circuit, parameters):  #parameters change the radian value
    #todo: Try to return the radian i.e parameter of 0 and 1 for the state as well!!
    q = circuit.qregs[0] # q is the quantum register where the info about qubits is stored
    circuit.rx(parameters[0], q[0]) # q[0] is our one and only qubit XD
    circuit.ry(parameters[1], q[0])
    print(circuit, "CIRCUIT with rx and ry q[0] parameters[0] and [1]: ", q[0],"..",parameters[0],"..",parameters[1])
    return circuit


H_gate = U2Gate(0, np.pi).to_matrix()
print("H_gate:")
print((H_gate * np.sqrt(2)).round(5))

Y_gate = U2Gate(0, np.pi/2).to_matrix()
print("Y_gate:")
print((Y_gate * np.sqrt(2)).round(5))



def vqe_circuit(parameters, measure):
    """
    Creates a device ansatz circuit for optimization.
    :param parameters_array: list of parameters for constructing ansatz state that should be optimized.
    :param measure: measurement type. E.g. 'Z' stands for Z measurement.
    :return: quantum circuit.
    """
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    circuit = QuantumCircuit(q, c)
    #print(circuit,"FIRST")

    # quantum state preparation
    circuit = quantum_state_preparation(circuit, parameters)

    # measurement
    if measure == 'Z':
        circuit.measure(q[0], c[0])
        #print(circuit.measure(q[0], c[0]),"@@")
    elif measure == 'X':
        circuit.u2(0, np.pi, q[0])
        circuit.measure(q[0], c[0])
        #print(circuit.measure(q[0], c[0]),"##")
    elif measure == 'Y':
        circuit.u2(0, np.pi/2, q[0])
        circuit.measure(q[0], c[0])
        #print(circuit.measure(q[0], c[0]),"$$")
    else:
        raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')
    #print(circuit, "Circuit with measurement added")
    print(circuit)
    return circuit



def quantum_module(parameters, measure):
    # measure
    if measure == 'I':
        return 1
    elif measure == 'Z':
        circuit = vqe_circuit(parameters, 'Z')
        #print(circuit, "here1")
    elif measure == 'X':
        circuit = vqe_circuit(parameters, 'X')
        #print(circuit, "here2")
    elif measure == 'Y':
        circuit = vqe_circuit(parameters, 'Y')
        #print(circuit, "here3")
    else:
        raise ValueError('Not valid input for measurement: input should be "I" or "X" or "Z" or "Y"')
    
    shots = 8192
    backend = BasicAer.get_backend('qasm_simulator')
    #print(backend," backend")
    job = execute(circuit, backend, shots=shots)
    #print(job, "job")
    result = job.result()
    #print(result, " result")
    counts = result.get_counts() #No of times we get 0 state and no of times we get 1 state
    #print(counts, "counts")
    
    # expectation value estimation from counts
    expectation_value = 0
    for measure_result in counts:
        #print(counts)
        #print(measure_result, " measure result ")
        sign = +1
        if measure_result == '1':
            sign = -1
        expectation_value += sign * counts[measure_result] / shots
        #print(expectation_value, " Expectation Value")
        
    return expectation_value


def pauli_operator_to_dict(pauli_operator):
    """
    from WeightedPauliOperator return a dict:
    {I: 0.7, X: 0.6, Z: 0.1, Y: 0.5}.
    :param pauli_operator: qiskit's WeightedPauliOperator
    :return: a dict in the desired form.
    """
    d = pauli_operator.to_dict()
    #print(pauli_operator, "Pauli operator, d = ", d)
    paulis = d['paulis']
    #print(paulis, "paulis")
    paulis_dict = {}

    for x in paulis:
        label = x['label']
        coeff = x['coeff']['real']
        paulis_dict[label] = coeff
    #print(paulis_dict, "paulis dict")
    return paulis_dict
pauli_dict = pauli_operator_to_dict(H)
#print(pauli_dict, "pauli dict H")


def vqe(parameters):
        
    # quantum_modules
    quantum_module_I = pauli_dict['I'] * quantum_module(parameters, 'I')
    #print(quantum_module_I, "QMI")
    quantum_module_Z = pauli_dict['Z'] * quantum_module(parameters, 'Z')
    #print(quantum_module_Z, "QMZ")
    quantum_module_X = pauli_dict['X'] * quantum_module(parameters, 'X')
    #print(quantum_module_X, "QMX")
    quantum_module_Y = pauli_dict['Y'] * quantum_module(parameters, 'Y')
    #print(quantum_module_Y, "QMY")
    
    # summing the measurement results
    classical_adder = quantum_module_I + quantum_module_Z + quantum_module_X + quantum_module_Y
    #print(classical_adder, " Classical Adder is the expectation value returned by the quantum system")
    
    return classical_adder

parameters_array = np.array([np.pi, np.pi])
#print(parameters_array, " Parameters array 1")
tol = 1e-3 # tolerance for optimization precision.


vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
print(vqe_result, " VQE Result")
print('The exact ground state energy is: {}'.format(reference_energy))
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))
print(vqe_result.x[0])
print("The estimated ground state is: [",vqe_result.x[0],",",vqe_result.x[1],"]")
#print(vqe_result.fun,"^^^^^^^^^^****")
#print(parameters_array, " Parameters array 2")