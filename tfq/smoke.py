"""TFQ smoke test: build the Skolik-style PQC and check gradients are live.

Mirrors qdqn-cartpole/src/models/vqc.py::_smoke_test -- batch of 8 states in,
[8, 2] out, and every parameter group must have a non-zero gradient. The point
is not that TFQ imports, it is that the differentiation path works end to end.
"""
import numpy as np, tensorflow as tf, tensorflow_quantum as tfq, cirq, sympy

n_qubits, n_layers, n_actions = 4, 5, 2
qubits = cirq.GridQubit.rect(1, n_qubits)

def one_qubit_rotation(q, symbols):
    return [cirq.rx(symbols[0])(q), cirq.ry(symbols[1])(q), cirq.rz(symbols[2])(q)]

def entangling_layer(qs):
    ops = [cirq.CZ(a, b) for a, b in zip(qs, qs[1:])]
    return ops + ([cirq.CZ(qs[0], qs[-1])] if len(qs) != 2 else [])

# Variational angles theta, encoding angles lambda*x -- exactly the tutorial's layout.
params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{n_qubits})')
inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

circuit = cirq.Circuit()
for l in range(n_layers):
    circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
    circuit += entangling_layer(qubits)
    circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))
circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i, q in enumerate(qubits))

ops = [cirq.Z(q) for q in qubits]
observables = [ops[0] * ops[1], ops[2] * ops[3]]   # one per action

symbols = [str(s) for s in list(params.flat) + list(inputs.flat)]
pqc = tfq.layers.ControlledPQC(circuit, observables, differentiator=None)

n_var, n_enc = len(list(params.flat)), len(list(inputs.flat))
theta = tf.Variable(tf.random.uniform((1, n_var), 0, 2*np.pi), name="theta")
lmbd  = tf.Variable(tf.ones((n_enc,)), name="lambda")
w     = tf.Variable(tf.ones((1, n_actions)), name="w")

batch = 8
states = tf.random.uniform((batch, n_qubits), -1., 1.)
empty = tfq.convert_to_tensor([cirq.Circuit() for _ in range(batch)])

# TFQ needs symbol values ordered to match `symbols`; sort exactly as the tutorial does.
idx = tf.constant([symbols.index(a) for a in sorted(symbols)])

with tf.GradientTape() as tape:
    tiled_theta = tf.tile(theta, multiples=[batch, 1])
    tiled_x = tf.tile(states, multiples=[1, n_layers])
    scaled = tf.math.multiply(tiled_x, tf.tile(tf.expand_dims(lmbd, 0), [batch, 1]))
    joined = tf.concat([tiled_theta, tf.keras.layers.Activation('tanh')(scaled)], axis=1)
    joined = tf.gather(joined, idx, axis=1)
    raw = pqc([empty, joined])                 # [batch, n_actions], each in [-1, 1]
    q = ((raw + 1) / 2) * w                    # tutorial's Rescaling layer
    loss = tf.reduce_sum(q)

grads = tape.gradient(loss, [theta, lmbd, w])
print(f"circuit variational params: {n_var}   encoding params: {n_enc}")
print(f"output shape: {tuple(q.shape)}")
assert tuple(q.shape) == (batch, n_actions), q.shape
for name, g in zip(["theta", "lambda", "w"], grads):
    assert g is not None, f"{name} gradient is None"
    norm = float(tf.norm(g))
    assert norm > 0, f"{name} gradient is dead"
    print(f"  {name:7s} grad norm: {norm:.6f}")
print("\nTFQ smoke test OK -- ControlledPQC forward + gradients live on all three groups")
