# Origin: https://github.com/koke-saka/quantum-machine-learning/blob/master/tf-quantum/QuantumCircuitLearning-TFQ.ipynb

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

import collections

# visualization tools
# %matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit, circuit_to_svg


# 学習データの作成 ------------------------------------------------------

## [x_min, x_max]のうち, ランダムにnum_x_train個の点をとって教師データとする.
x_min = - 1.; x_max = 1.;
num_x_train = 200

## 学習したい1変数関数
#func_to_learn = lambda x: np.cos(x*np.pi)
func_to_learn = lambda x: x**2-0.5

random_seed = 1
np.random.seed(random_seed)

#### 教師データを準備
x_train = x_min + (x_max - x_min) * np.random.rand(num_x_train)
y_train = func_to_learn(x_train)

# 現実のデータを用いる場合を想定し、きれいなsin関数にノイズを付加
mag_noise = 0.025
y_train = y_train + mag_noise * np.random.randn(num_x_train)

plt.plot(x_train, y_train, "o"); plt.show()

def divide_train_test(data,label, test_ratio=0.3):
    shuffled = np.random.permutation(len(data))
    test_size = int(len(data)*test_ratio)
    test_index = shuffled[:test_size]
    train_index = shuffled[test_size:]
    return data[train_index],label[train_index],data[test_index],label[test_index]

x_train,y_train,x_test,y_test=divide_train_test(x_train,y_train,test_ratio=0.25)


# 入力状態の作成 ------------------------------------------------------

def convert_to_circuit(x):
    """Encode truncated classical image into quantum datapoint."""
    y = np.arcsin(x)
    z = np.arccos(x**2)
    qubits = cirq.GridQubit.rect(5, 1)
    circuit = cirq.Circuit()
    for i in range(5):
        circuit.append(cirq.ry(y).on(qubits[i]))
        circuit.append(cirq.rz(z).on(qubits[i]))
    return circuit

x_train_circ = [convert_to_circuit(x) for x in x_train]
x_test_circ = [convert_to_circuit(x) for x in x_test]

import uuid
def save_text(src: str, dest_path=None, file_ext="svg"):
    if dest_path is None:
        dest_path = str(uuid.uuid4()) + ".{}".format(file_ext)
    with open(dest_path, 'w', encoding='utf-8') as f: 
        f.write(src)

def dump_circuit(circuit):
    save_text(circuit_to_svg(circuit))

# SVGCircuit(x_train_circ[0])
dump_circuit(x_train_circ[0])

x_train[0],np.arcsin(x_train[0])/np.pi

x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

# パラメトリック量子回路（ニューラルネット）の作成 ---------------

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)
            
    def add_layer_single(self,circuit,gate,prefix):
        symbol = sympy.Symbol(prefix + '-' + str(0))
        circuit.append(gate(symbol).on(self.readout))
    
    def add_entangler(self,circuit,len_qubit):
        circuit.append(cirq.CZ(self.readout,self.data_qubits[0]))
        for i in range(len_qubit-1):
            circuit.append(cirq.CZ(self.data_qubits[i],self.data_qubits[(i+1)%len_qubit]))
        circuit.append(cirq.CZ(self.readout,self.data_qubits[-1]))

def create_quantum_model(c_depth=3):
    data_qubits = cirq.GridQubit.rect(5,1)
    readout = cirq.GridQubit(-1,-1)
    circuit = cirq.Circuit()
    
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout = readout
    )
    
    for i in range(3):
        builder.add_entangler(circuit,5)
        builder.add_layer(circuit, gate = cirq.XX, prefix='xx'+str(i))
        builder.add_layer(circuit, gate = cirq.ZZ, prefix='zz'+str(i))
        builder.add_layer(circuit, gate = cirq.XX, prefix='xx1'+str(i))
        builder.add_layer_single(circuit, gate = cirq.rz, prefix='z1'+str(i))
        builder.add_layer_single(circuit, gate = cirq.rx, prefix='x1'+str(i))
        builder.add_layer_single(circuit, gate = cirq.rz, prefix='z2'+str(i))
   
    return circuit, cirq.Z(readout)

model_circuit, model_readout = create_quantum_model()

# SVGCircuit(model_circuit)
dump_circuit(model_circuit)

model_readout


# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout),
])

model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae'])

print(model.summary())

EPOCHS = 100
BATCH_SIZE = 50

qnn_history = model.fit(
      x_train_tfcirc, y_train,
      batch_size=25,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc,y_test)
)

model.evaluate(x_test_tfcirc,y_test)

y_pred = model.predict(x_test_tfcirc)

x_test.shape

y_pred.shape


plt.plot(x_test,y_pred,"o",label="pred")
plt.plot(x_test,y_test,"xr",label="test")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()
plt.savefig('quadratic_equation_tfq_result.png')





