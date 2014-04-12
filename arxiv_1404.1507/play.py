#!/usr/bin/env python
# coding: utf-8

""" Plays around with Wiesner's quantum money scheme and this papers attack on
    it.

    All of this is in qubits.
    """

# HACK/WARNING: Numpy and random use different seeds.
import random
import numpy as np

states = {
    "0": np.array([[1], [0]]),
    "1": np.array([[0], [1]]),
    "+": np.array([[1], [1]])/np.sqrt(2),
    "-": np.array([[1], [-1]])/np.sqrt(2),
}

H = np.array([[1,1],[1,-1]])/np.sqrt(2)

# Our serial number database. is the map:
#
#   s -> bases
#
# say
#   "123" -> ["0", "0", "0", "0", "0", "0", "0"]
#
bankDatabase = {}

# General bits
# -------------------------------

def weightedChoice (values, probabilities):
    """ Returns a weight choice from the values according to the probability 
        distribution.
        
        >>> weightedChoice( ["a", "b"], [ 0, 1 ] )
        'b'

        >>> weightedChoice( ["a", "b"], [ 1, 0 ] )
        'a'

        Used to return the state based on the square of the amplitudes in some
        basis.
        """

    assert abs(sum(probabilities)-1) < 1e-10, "Probs sum to: {0:f}".format(sum(probabilities))

    size  = 1000 # Increase to improve the accuracy.
    bins  = np.add.accumulate(probabilities)
    space = np.array(values)[np.digitize(np.random.random_sample(size), bins)]
    return random.choice(space)


def measure (inState, basis="0,1", qubits=None):
    """ Measures the given state in the given basis. Will return a new state.

        >>> measure( listToState(["0", "0"]), basis="0,1", qubits=[1] )
        (['0'], array([[ 1.],
               [ 0.],
               [ 0.],
               [ 0.]]))

        >>> measure( listToState(["1", "0"]), basis="0,1", qubits=[1] )
        (['1'], array([[ 0.],
               [ 0.],
               [ 1.],
               [ 0.]]))

        >>> measure( listToState(["0", "0"]) )
        (['0', '0'], array([[ 1.],
               [ 0.],
               [ 0.],
               [ 0.]]))

        >>> measure( listToState(["+", "+"]),  basis="+,-", qubits=[1] )
        (['+'], array([[ 0.5],
               [ 0.5],
               [ 0.5],
               [ 0.5]]))
        
        >>> measure( listToState(["+", "0"]),  basis="+,-", qubits=[1] )
        (['+'], array([[ 0.70710678],
               [ 0.        ],
               [ 0.70710678],
               [ 0.        ]]))
        
        >>> measure( listToState(["+", "0"]),  basis="0,1", qubits=[2] )
        (['0'], array([[ 0.70710678],
               [ 0.        ],
               [ 0.70710678],
               [ 0.        ]]))

        >>> np.random.seed(1) # Requires a seed as the first component of a is nondetermininstic
        >>> measure( listToState(["+", "0"]),  basis="0,1", qubits=[1, 2] )
        (['1', '0'], array([[ 0.70710678],
               [ 0.        ],
               [ 0.70710678],
               [ 0.        ]]))

        Returns a tuple consisting of: (measurementOutcomes, newState).

        Note: Qubits are indexed in a way that is compatibile with Mathematica.
        Note: For convenience, measurement always happens
    """
    assert basis in ["0,1", "+,-"]

    # Some local vairables.
    state = inState
    dim   = int(np.log2(len(state)))

    if not qubits: # no qubits? all qubits.
        qubits = xrange(1, dim+1)

    basisChanger = np.identity(len(state))

    if basis == "+,-":
        # Hadamard the bits we're measuring, then just measure them in the
        # computational basis.
        
        H_or_I = lambda x: H if (x in qubits) else np.identity(2)
        
        allHadamards = reduce(np.kron, [H_or_I(x) for x in xrange(1, dim+1)], 1)
        # state = np.dot( allHadamards, state )
        basisChanger = allHadamards

    # We always measure in the computational basis.
    measurementBasis = [ listToState(["0"]), listToState(["1"]) ]

    outcomes = []
    for qubitPosition in qubits:
        # To be used with "weightedChoice"
        options = []
        probs   = []

        # Okay, so for this position, we should like to build I P_{e_i} I, for
        # each projector P_{e_i}.

        for (i, ei) in enumerate(measurementBasis):
            proj = reduce(np.kron, [ np.identity(2**(qubitPosition-1)),
                np.dot(ei, np.conj(np.transpose(ei))),
                np.identity(2**(dim-qubitPosition)) ], 1)

            # Switch proj
            proj = np.dot( np.conj(np.transpose(basisChanger)), np.dot( proj, basisChanger ) )

            vec  = np.dot( proj, state )
            prob = np.dot( np.conj(np.transpose(vec)), vec )
            #
            # vec  = np.dot(basisChanger, vec)

            # import pdb
            # pdb.set_trace()

            options.append( {"i": str(i), "vec": vec} )
            probs.append( prob.flatten()[0] )

        data = weightedChoice(options, np.array(probs))
        
        # 
        # Great. We can now update the state, we also probably need to
        # normalise it.
        #

        state = data["vec"]
        state = state/np.linalg.norm(state)

        # import pdb
        # pdb.set_trace()

        # Lazily relabel if we were asked to.
        if basis == "+,-": basisLabel = {"0": "+", "1": "-"}[data["i"]]
        else:              basisLabel = data["i"]

        outcomes.append(basisLabel)

    # state = np.dot( basisChanger, state )

    # Return the measurement outcomes for the qubits, and the resulting state.
    return (outcomes, state)


def listToState (key):
    """ Builds a quantum state from the "key" specifying the state.
        >>> listToState(["1", "0"])
        array([[0],
               [0],
               [1],
               [0]])
        """

    # Build the state with fold(...)
    keyState = reduce(np.kron, [states[x] for x in key], 1)
    return keyState


# Attacker-bits
# -------------------------------

# def naivelyForge ():



# Bank-related bits.
# -------------------------------

def deposit ( (s, keyState) ):
    """ Returns True if the money can be deposited. False otherwise.
        Furthermore, if we will actually notify the police if we determine
        that the money is not legitimate.

        >>> (s, key) = generateMoneyData(20, 2)
        >>> keyState = listToState(key)
        >>> deposit( (s, keyState) )
        True
        """
    return False


def generateMoneyData (amount, seed=None):
    """ Returns a tuple (s, k_s) where s is the serial number, and k_s is the
        key that will be used to build the money state.

        >>> generateMoneyData(20, 2)
        (123, ['-', '0', '0', '-', '+', '+', '1'])
        """

    if seed: random.seed(seed)

    # Keysize.
    n = 7
    s = random.randint(0, 2**7)

    # Let's generate a 7-bit key from the set of states {0,1,+,-}.
    alphabet = ["0", "1", "+", "-"]
    key = [ random.choice(alphabet) for k in xrange(n) ]

    # Save this key.
    bankDatabase[s] = key

    return (s, key)


def getMoney (amount):
    """ Returns (s,|k_s>).
        """
    (s, key) = generateMoneyData(amount)
    keyState = keyState(key)

    return (s, keyState)



def test_sadness_1 ():
    (a, b) = measure( listToState(["+", "0"]),  basis="0,1", qubits=[1,2] )
    assert a[1] == '0'

    # What can "b" be?
    #
    #
    # assert all(b == np.array([[ 0.70710678],
    #            [ 0.        ],
    #            [ 0.70710678],
    #            [ 0.        ]]))
