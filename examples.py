from math import sqrt
from cmath import exp, pi

from qsim import apply, gate, state, SWAP

H = [[1 / sqrt(2),  1 / sqrt(2)],
     [1 / sqrt(2), -1 / sqrt(2)]]

CNOT = [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]]

def CPHASE(radians: float) -> gate:
    return [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, exp(1j * radians)]]

def bell(p: list, q: list):
    return [['GATE', H, p],
            ['GATE', CNOT, q]]

def ghz(n: int):
    result = [['GATE', H, 0]]
    for i in range(n - 1):
        result.append(['GATE', CNOT, i, i + 1])
    return result

def qft(qubits: list):
    def bit_reversal(qubits: list) -> list:
        n = len(qubits)
        result = []
        for i in range(n // 2): # skipped for n < 2
            qs = qubits[i]
            qe = qubits[-i - 1]
            result.append(['GATE', SWAP, qs, qe])
        return result

    def _qft(qubits: list) -> list:
        q, *qs = qubits
        if not qs:
            return [['GATE', H, q]]

        n = len(qs)
        cR = []
        for i in range(n):
            qi = qs[i]
            angle = pi / (2 ** (n - i))
            cR.append(['GATE', CPHASE(angle), q, qi])

        result = _qft(qs)
        result.extend(cR)
        result.append(['GATE', H, q])
        return result

    ft = _qft(qubits)
    br = bit_reversal(qubits)
    return ft + br
