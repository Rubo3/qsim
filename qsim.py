# state = [c_0, c_1, c_2, ..., c_{2^n - 1}]
# gate = [row1, row2, ..., row_{2^n}]

from functools import reduce
from math import log2
from random import uniform
from typing import List

state = List[complex]
gate = List[List[complex]]

I = [[1,0],
     [0,1]]

SWAP = [[1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]]

class Machine:
    def __init__(self, s: state, measurement_register: int):
        self.state = s
        self.measurement_register = measurement_register

# initialise a quantum state as 𝜓 = |0...0⟩
def make_quantum_state(n: int) -> state:
    state = [0] * (2 ** n)
    state[0] = 1
    return state

def dimension_qubits(length: int) -> int:
    return int(log2(length))

def apply(gate: gate, s: state) -> state:
    assert all(len(s) == len(row) for row in gate)

    result = [0] * len(gate)
    for i in range(len(gate)):
        for j in range(len(gate[0])):
            result[i] += gate[i][j] * s[j]
    return result

def compose(A: gate, B: gate) -> gate:
    assert all(len(B) == len(row) for row in A)

    m = len(A)
    n = len(A[0]) # == len(B)
    o = len(B[0])
    result = [[0] * o for _ in range(m)]
    for i in range(m):
        for j in range(o):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

def sample(s: state) -> int:
    r = uniform(0, 1) # 0 <= n <= 1
    for i in range(len(s)):
        probability = abs(s[i]) ** 2
        if r < probability:
            return i
        r -= probability

def collapse(s: state, i: int):
    assert(0 <= i < len(s))

    for j in range(len(s)):
        s[j] = 0
    s[i] = 1

def observe(machine: Machine) -> Machine:
    state = machine.state
    i = sample(state)
    collapse(state, i)
    machine.measurement_register = i
    return machine

def kronecker_mul(A: gate, B: gate) -> gate:
    m = len(A)
    n = len(A[0])
    p = len(B)
    q = len(B[0])
    result = [[0] * (n * p) for _ in range(m * q)]
    for i in range(m * p):
        for j in range(n * q):
            result[i][j] += A[i // p][j // q] * B[i % p][j % q]
    return result

def kronecker_exp(U: gate, n: int) -> gate:
    if n < 1:
        return [[1]]
    if n == 1:
        return U
    return kronecker_mul(kronecker_exp(U, n - 1), U)

def lift(U: gate, i: int, n: int) -> gate:
    left = kronecker_exp(I, n - i - dimension_qubits(len(U[0])))
    right = kronecker_exp(I, i)
    result = kronecker_mul(left, kronecker_mul(U, right))
    return result

# permutation to transposition
def perm2trans(permutation: list[int]) -> list[int]:
    swaps = []
    for dest in range(len(permutation)):
        src = permutation[dest]
        while src < dest:
            src = permutation[src]
        if src < dest:
            swaps.append((src, dest))
        elif src > dest:
            swaps.append((dest, src))
    return swaps

# transpositions to adjacent transpositions
def trans2adj(transpositions: list[int]) -> list[int]:
    def expand_consecutive(c):
        if c[1] - c[0] == 1:
            return [c[0]]
        trans = list(range(c[0], c[1]))
        return trans + list(reversed(trans[:-1]))

    result = [i for c in transpositions for i in expand_consecutive(c)]
    return result

def apply_1Q(s: state, U: gate, qubit: int) -> state:
    lifted_U = lift(U, qubit, dimension_qubits(len(s)))
    result = apply(lifted_U, s)
    return result

def apply_nQ(s: state, U: gate, qubits: list[int]):
    n = dimension_qubits(len(s))

    # swap qubit i with qubit i + 1
    def swap(i: int) -> gate:
        result = lift(SWAP, i, n)
        return result

    # transpositions to operator
    def trans2op(transpositions):
        result = swap(transpositions[0])
        for t in transpositions[1:]:
            result = compose(result, swap(t))
        return result

    U01 = lift(U, 0, n)

    from_space = list(reversed(qubits)) + [i for i in range(n) if i not in qubits]
    trans = perm2trans(from_space)
    adj = trans2adj(trans)
    to_from = trans2op(adj)
    from_to = trans2op(adj[::-1])

    Upq = compose(to_from, compose(U01, from_to))

    result = apply(Upq, s)
    return result

# `qubits` is the list of indices of the qubits you want to operate on
def apply_gate(s: state, U: gate, qubits: list[int]):
    assert(len(qubits) == dimension_qubits(len(U[0])))

    if len(qubits) == 1:
        return apply_1Q(s, U, qubits[0])
    return apply_nQ(s, U, qubits)

def run(qprog, machine):
    for instruction, *payload in qprog:
        if instruction == 'GATE':
            gate, *qubits = payload
            machine.state = apply_gate(machine.state, gate, qubits)
        elif instruction == 'MEASURE':
            machine = observe(machine)
    return machine
