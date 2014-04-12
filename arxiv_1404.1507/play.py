#!/usr/bin/env python
# coding: utf-8

""" Plays around with Wiesner's quantum money scheme and this papers attack on
    it.

    All of this is in qubits.
    """

import random, numpy

states = {
    "0": numpy.array([[1], [0]]),
    "1": numpy.array([[0], [1]]),
    "+": numpy.array([[1], [1]])/numpy.sqrt(2),
    "-": numpy.array([[1], [-1]])/numpy.sqrt(2),
}

H = numpy.array([[1,1],[1,-1]])/numpy.sqrt(2)

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

def measure (inState, basis="0,1", qubits=None):
    """ Measures the given state in the given basis. Will return a new state.
        >>> measure( listToState(["0", "0"]), basis="0,1", qubits=[1,2] )
        ( [0, 0], ... )
        
        >>> measure( listToState(["0", "0"]) )
        ( [0, 0], ... )

        >>> measure( listToState(["+", "+"]),  basis="+,-" )
        ( [0, 0], ... )

        Returns a tuple consisting of: (measurementOutcomes, newState).
    """
    state = inState
    assert basis in ["0,1", "+,-"]

    dim = int(numpy.log2(len(state)))

    if basis == "+,-":
        # Hadamard the entire thing.
        allHadamards = reduce(np.kron, [H for x in xrange(dim)], 1)
        state = numpy.prod( allHadamards, state )

    # Build the 2^dim basis elements by string manipulation.
    measurementBasis = [ listToState( list(bin(x)[2:].zfill(dim)) ) 
            for x in xrange(len(state)) ]

    # Now we can do a POVM.

    povm = sum( numpy.dot( v, numpy.conj(numpy.transpose(v)) ) for v in
            measurementBasis )

    # Check that it actually is a POVM.
    assert numpy.all(povm**2 == numpy.identity(len(state)))



    # 1. We want to construct a basis as large as the input state.
    pass


def listToState (key):
    """ Builds a quantum state from the "key" specifying the state.
        >>> listToState(["1", "0"])
        array([[0],
               [0],
               [1],
               [0]])
        """

    # Build the state with fold(...)
    keyState = reduce(numpy.kron, [states[x] for x in key], 1)
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

