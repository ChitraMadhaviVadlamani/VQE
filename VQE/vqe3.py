#IX only

import numpy as np
from random import random
from scipy.optimize import minimize
#from qiskit.quantum_info.operators.operator import tensor
from qiskit import QuantumCircuit
#print(bottom^top,"@@") tensor product

from qiskit import QuantumCircuit
top = QuantumCircuit(1)
top.x(0)
bottom = QuantumCircuit(2)
bottom.cry(0.2, 0, 1)
tensored = bottom.tensor(top)
print(tensored.draw())

from qiskit import *
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver

def hamiltonian_operator(a, b, c):
    """
    Creates a*I + b*Z + c*X + d*Y pauli sum 
    that will be our Hamiltonian operator.
    H = aXY
    """
    #since no i the imaginary coefficients are set to 0
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": 1}, "label": "IX"}  #a is the weight and I is the Pauli Operator
                   ]
    }
    #print(pauli_dict,"Paulis Dict")
    #print(WeightedPauliOperator.from_dict(pauli_dict),"WPO######")
    return WeightedPauliOperator.from_dict(pauli_dict)
    #return WeightedPauliOperator.from_dict(pauli_dict,"Pauli_dict 1")


scale = 10
a, b, c = (scale*random(), scale*random(), scale*random())
#print(a," ", b, " ", c, " ",d)
H = hamiltonian_operator(a, b, c) #method to make hamiltonians divided
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
    #print(circuit, "CIRCUIT with rx and ry q[0] parameters[0] and [1]: ", q[0],"..",parameters[0],"..",parameters[1])
    return circuit


H_gate = U2Gate(0, np.pi).to_matrix()
print("H_gate:")
print((H_gate * np.sqrt(2)).round(5))

Y_gate = U2Gate(0, np.pi/2).to_matrix()
print("Y_gate:")
print((Y_gate * np.sqrt(2)).round(5))





def vqe_circuit(parameters, measure):
    measure = "IX"
    """
    Creates a device ansatz circuit for optimization.
    :param parameters_array: list of parameters for constructing ansatz state that should be optimized.
    :param measure: measurement type. E.g. 'Z' stands for Z measurement.
    :return: quantum circuit.
    """
    q = QuantumRegister(1)
    #c = ClassicalRegister(1)
    circuit = QuantumCircuit(q)
    q2 = QuantumRegister(1)
    c2 = ClassicalRegister(1)
    circuit1 = QuantumCircuit(q2)
    circuit = quantum_state_preparation(circuit, parameters)
    circuit1 = quantum_state_preparation(circuit1, parameters)
    print("KKKK",circuit,"@@",circuit1,"HELLO")
    
    if measure == 'XY':
        print("HERRE")
        
        circuit.u2(0, np.pi/2, q[0])
        circuit1.u2(0, np.pi, q2[0])
        tensored = circuit1.tensor(circuit)
        tensored.measure_all(q[0],q2[0])
        #print(tensored,"PLEASE WORK")
        #circuit.measure(q[0], c[0])
        #print(circuit.measure(q[0], c[0]),"$$")
    
    elif measure == "IX":
        print("IX HERE")
        circuit1.u2(0, np.pi, q2[0])
        tensored = circuit1.tensor(circuit)
        tensored.measure_all(q[0],q2[0])
        print(tensored,"PLEASE  IX")

    
    else:
        raise ValueError('Not valid input for measurement: input should be "IX" or "XY" or "IZ"')
    #print(circuit, "Circuit with measurement added")
    return tensored



def quantum_module(parameters, measure):
    measure = "IX"
    # measure
    if measure == 'IX':
        circuit = vqe_circuit(parameters, 'IX')
        #print(circuit, "here1")
    #elif measure == 'XY':
     #   circuit = vqe_circuit(parameters, 'XY')
        #print(circuit, "here3")
    else:
        raise ValueError('Not valid input for measurement: input should be "IX" or "XY" or "IZ"')
    
    shots = 8192
    backend = BasicAer.get_backend('qasm_simulator')
    #print(backend," backend")
    #print(circuit,"@@")
    job = execute(circuit, backend, shots=shots)
    #print(job, "job")
    result = job.result()
    #print(result, " result")
    counts = result.get_counts() #No of times we get 0 state and no of times we get 1 state
    print(counts, "counts")
    
    # expectation value estimation from counts
    expectation_value = 0
    print(counts,"outside")
    for measure_result in counts:
        print(counts,"inside")
        print(measure_result, " measure result ")
        sign = +1
        if measure_result == '01' or '10':
            sign = -1
        expectation_value += sign * counts[measure_result] / shots  #first it gives probability of 0 adds it to expectation_value = 0; then it finds the prob 1 and then it adds it to prob 0 
        print(expectation_value," EV")
        #print(sign, "@@",counts[measure_result],"!!")
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
    quantum_module_IX = pauli_dict['IX'] * quantum_module(parameters, 'IX')
    
    #quantum_module_XY = pauli_dict['XY'] * quantum_module(parameters, 'XY')
    #return quantum_module_XY +quantum_module_IX
    return quantum_module_IX
    #return classical_adder

parameters_array = np.array([np.pi, np.pi,np.pi,np.pi])
#print(parameters_array, " Parameters array 1")
tol = 1e-3 # tolerance for optimization precision.
#tol = 900000000

vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
#print(vqe_result, " VQE Result")
print('The exact ground state energy is: {}'.format(reference_energy))
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))
#print(vqe_result.x[0])
print("The estimated ground state is: [",vqe_result.x[0],",",vqe_result.x[1],"]")
#print(vqe_result.fun,"^^^^^^^^^^****")
#print(parameters_array, " Parameters array 2")