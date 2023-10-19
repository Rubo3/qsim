# qsim

A self-contained Python implementation of Robert Smith's [*A tutorial quantum interpreter in 150 lines of Lisp*](https://www.stylewarning.com/posts/quantum-interpreter/).

```python
from qsim import Machine, make_quantum_state, run
from examples import H

m = Machine(make_quantum_state(1), 0)
qprog = [
    ['GATE', H, 0],
    ['MEASURE']
]

flips = []
for _ in range(10):
    m = run(qprog, m)
    flips.append(m.measurement_register)
print(flips)
```

## License

This software is free and open-source and distributed under the MIT License.
