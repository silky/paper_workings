notes
==

if we entangle the key, this attack somewhat-trivally doesn't work. the
problem is that both the tests for X, -X fail, and then we decide to measure
in the computational basis. This breaks the key.

so we need to not do that.

ideas:

  * can we find out which qubits are entangled without destroying them?
  * what does this say about the possible states we can clone?
