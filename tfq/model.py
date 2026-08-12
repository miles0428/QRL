"""Skolik-style PQC Q-function in TensorFlow Quantum.

A FAITHFUL reproduction of the TFQ tutorial "Parametrized Quantum Circuits for
Reinforcement Learning", Section 3, which implements Skolik, Jerbi & Dunjko
(2022) https://arxiv.org/abs/2103.15084 -- the same paper the Qiskit
implementation in src/ follows.

WHY A SECOND IMPLEMENTATION
---------------------------
The Qiskit arm reaches greedy 500 but does not hold it: its policy oscillates
between optimal and collapsed (see 報告.md section F). Comparing the two arms
separates the two possible causes -- a bug in our implementation, versus this
hyperparameter set simply behaving this way. That only works if this file is a
faithful reproduction, so where the tutorial and our implementation disagree,
THIS FILE FOLLOWS THE TUTORIAL, deliberately, even where our choice looks more
principled. Each such point is marked "DIVERGES FROM src/".

The differences, all of them:

  encoding gate     RX here; RY in src/
  rotations/layer   RX,RY,RZ here (3); RY,RZ in src/ (2)
  variational layers  n_layers+1 = 6 here; n_layers = 5 in src/
  entangler         CZ ring here; CNOT ring in src/
  input scaling     lambda is per (layer, qubit), 20 values; 4 in src/
  input squashing   tanh(lambda * x) here; none in src/
  state preparation raw observation here; normalize_observation() in src/
  output            (<O>+1)/2 * w, i.e. [0,1]; <O> * w, i.e. [-1,1], in src/

Trainable parameters: 72 variational + 20 lambda + 2 w = 94, versus 46 in src/.
The parameter-matched classical baseline (44 params) is matched to src/, NOT to
this, so a parameter-count comparison against mlp_baseline is not apples to
apples for this arm. Say so wherever it is reported.
"""

from __future__ import annotations

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq


def one_qubit_rotation(qubit, symbols):
    """RX, RY, RZ on one qubit -- a general single-qubit rotation."""
    return [
        cirq.rx(symbols[0])(qubit),
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit),
    ]


def entangling_layer(qubits):
    """CZ ring. (DIVERGES FROM src/, which uses a CNOT ring.)"""
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += [cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else []
    return cz_ops


def generate_circuit(qubits, n_layers):
    """Data re-uploading circuit: n_layers x [variational, entangle, encode],
    then a final variational layer.

    Note the ordering: the encoding gates come AFTER the entangler within a
    layer, and there is one more variational layer than there are encoding
    layers. Returns (circuit, variational_symbols, input_symbols) with the
    symbols as flat lists.
    """
    n_qubits = len(qubits)

    params = sympy.symbols(f"theta(0:{3 * (n_layers + 1) * n_qubits})")
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    inputs = sympy.symbols(f"x(0:{n_layers})" + f"_(0:{n_qubits})")
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    circuit = cirq.Circuit()
    for layer in range(n_layers):
        circuit += cirq.Circuit(
            one_qubit_rotation(q, params[layer, i]) for i, q in enumerate(qubits)
        )
        circuit += entangling_layer(qubits)
        circuit += cirq.Circuit(cirq.rx(inputs[layer, i])(q) for i, q in enumerate(qubits))

    circuit += cirq.Circuit(
        one_qubit_rotation(q, params[n_layers, i]) for i, q in enumerate(qubits)
    )

    return circuit, list(params.flat), list(inputs.flat)


class ReUploadingPQC(tf.keras.layers.Layer):
    """The PQC with trainable variational angles `theta` and input scaling `lmbd`.

    SYMBOL ORDERING is the trap here, and it is the same class of bug as the
    Qiskit arm's (qiskit sorts circuit parameters alphabetically by default).
    `ControlledPQC` expects symbol values in the circuit's *sorted symbol* order,
    so `self.indices` is the permutation from "thetas then inputs" into sorted
    order. Getting it wrong silently trains a different circuit -- the forward
    pass looks completely normal.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True,
            name="thetas",
        )

        # One scaling per (layer, qubit): the same observation component is
        # re-uploaded at every layer and each upload is scaled independently.
        # (DIVERGES FROM src/, where lam has one value per qubit, shared across
        # layers.)
        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas")

        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])

        # Each observation component is repeated once per layer, then scaled.
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        # tanh keeps the encoding angle in [-1, 1] however large lambda grows.
        # (DIVERGES FROM src/, which has no squashing and instead normalizes the
        # observation before scaling.)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])


class Rescaling(tf.keras.layers.Layer):
    """(<O> + 1) / 2 * w.

    The (x+1)/2 maps the expectation value from [-1, 1] to [0, 1] BEFORE the
    trainable weight is applied. This matters more than it looks: CartPole
    returns are all non-negative and Q* at gamma=0.99 is ~99, so on [0, 1] a
    weight near 100 suffices. src/ multiplies w against [-1, 1] instead, which
    is why its w drifts to 300-500 -- representing Q~100 from <O>~0.2 requires
    w~500. Same expressivity, very different gradient scale.
    """

    def __init__(self, input_dim):
        super().__init__(name="Q-values")
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1, input_dim)), dtype="float32", trainable=True,
            name="obs-weights",
        )

    def call(self, inputs):
        return tf.math.multiply(
            (inputs + 1) / 2, tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0)
        )


def build_q_model(n_qubits: int = 4, n_layers: int = 5, n_actions: int = 2):
    """Keras model: observation [B, n_qubits] -> Q-values [B, n_actions].

    Observables are two-qubit Z products, one per action: Z0*Z1 and Z2*Z3, so
    each observable reads half the register. Same choice as src/ (whose Qiskit
    strings "IIZZ"/"ZZII" denote the same two operators; the action assignment
    is mirrored, which is symmetric and not a bug).
    """
    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [ops[0] * ops[1], ops[2] * ops[3]]

    if len(observables) != n_actions:
        raise ValueError(f"{len(observables)} observables for {n_actions} actions")

    input_tensor = tf.keras.Input(shape=(n_qubits,), dtype=tf.dtypes.float32, name="input")
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation="tanh")([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name="observables-policy")
    model = tf.keras.Model(inputs=[input_tensor], outputs=process(re_uploading_pqc))
    return model


# Trainable-variable index of each parameter group, for the three-optimizer
# setup. Asserted at startup in train.py rather than trusted: a silent
# reordering would apply the output-scaling learning rate (0.1) to the circuit
# weights (0.001), which trains but trains wrongly.
W_VAR, W_IN, W_OUT = 0, 1, 2


def parameter_group_report(model) -> str:
    names = [v.name for v in model.trainable_variables]
    shapes = [tuple(v.shape) for v in model.trainable_variables]
    total = sum(int(np.prod(s)) for s in shapes)
    lines = [f"trainable variables ({total} scalars total):"]
    for i, (n, s) in enumerate(zip(names, shapes)):
        tag = {W_VAR: "variational", W_IN: "input scaling", W_OUT: "output scaling"}.get(i, "?")
        lines.append(f"  [{i}] {n:24s} {str(s):12s} {tag}")
    return "\n".join(lines)
