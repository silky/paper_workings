#!/usr/bin/env python
# coding: utf-8

""" Plays around with Wiesner's quantum money scheme and this papers attack on
    it.

    All of this is in qubits.
    """


# HACK/WARNING: Numpy and random use different seeds.
import random
import copy
import numpy as np

states = {
    "0": np.array([[1], [0]]),
    "1": np.array([[0], [1]]),
    "+": np.array([[1], [1]])/np.sqrt(2),
    "-": np.array([[1], [-1]])/np.sqrt(2),
}

I = np.identity(2)
X = np.array([[0,1],[1,0]])
H = np.array([[1,1],[1,-1]])/np.sqrt(2)
n = 3 # They keysize

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

def ct (array):
    """ Conjugate-transpose.  """
    return np.conj(np.transpose(array))


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

        Returns a tuple consisting of: (measurementOutcomes, newState).

        Note: Qubits are indexed in a way that is compatibile with Mathematica.
        Note: For convenience, measurement always happens

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
        >>> random.seed(1) # Requires a seed as the first component of a is nondetermininstic
        >>> measure( listToState(["+", "0"]),  basis="0,1", qubits=[1, 2] )
        (['1', '0'], array([[ 0.],
               [ 0.],
               [ 1.],
               [ 0.]]))

    """
    assert basis in ["0,1", "+,-"]
    assert abs(np.linalg.norm(inState) - 1) < 1e-10, "Norm not 1."

    # Some local vairables.
    state = inState
    dim   = int(np.log2(len(state)))

    if not qubits: # no qubits? all qubits.
        qubits = xrange(1, dim+1)

    basisChanger = np.identity(len(state))

    if basis == "+,-":
        # Hadamard the bits we're measuring, then just measure them in the
        # computational basis. So build a thing that changes basis.
        
        H_or_I = lambda x: H if (x in qubits) else I
        
        allHadamards = reduce(np.kron, [H_or_I(x) for x in xrange(1, dim+1)], 1)
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

            options.append( {"i": str(i), "vec": vec} )
            probs.append( prob.flatten()[0] )

        data = weightedChoice(options, np.array(probs))
        
        # 
        # Great. We can now update the state, we also probably need to
        # normalise it.
        #

        state = data["vec"]
        state = state/np.linalg.norm(state)

        # Lazily relabel if we were asked to.
        if basis == "+,-": basisLabel = {"0": "+", "1": "-"}[data["i"]]
        else:              basisLabel = data["i"]

        outcomes.append(basisLabel)

    # assert abs(np.linalg.norm(state) - 1) < 1e-10, "Norm not 1."

    # Return the measurement outcomes for the qubits, and the resulting state.
    return (outcomes, state)


def listToState (key):
    """ Builds a quantum state from the "key" specifying the state.
        >>> listToState(["1", "0"])
        array([[0],
               [0],
               [1],
               [0]])

        >>> listToState(["+", "-"])
        array([[ 0.5],
               [-0.5],
               [ 0.5],
               [-0.5]])

        >>> listToState(["+", "+"])
        array([[ 0.5],
               [ 0.5],
               [ 0.5],
               [ 0.5]])
        """

    # Build the state with fold(...)
    keyState = reduce(np.kron, [states[x] for x in key], 1)
    return keyState


# Attacker-bits
# -------------------------------

def bombTester (state):
    """ Measures the state and checks if it was a bomb or not.
    """

    # the "+1" is because we know that the control qubit is in the first
    # position.
    
    (a, state) = measure(state, basis="0,1", qubits=[1 +1])

    didExplode = lambda x: x == "1"
    exploded   = didExplode(a[0])

    return (exploded, state)


def isItABomb (bombOperator, bombState, qubit, tester, N=None):
    """ Implementation of the Elitzur-Vaidman "bomb-quality-tester". Or perhaps
        actually just an implementation of the circuit in Figure 1 of the
        paper.

        bombOperator:   A single-qubit operation, whose measurement outcome we
                        will be interested in.

        bombState:      The state which we are applying the bombOperator to.
                        Note that we will only apply to the specified qubit.

        qubit:          The qubit, inside the bomb state, onto which we shall
                        enact the 'bombOperator'.

        tester:         state -> (Bool, state)

                        A function that takes a state and returns a state.
                        This function should also explode or send the caller
                        to jail if it notices a particular outcome for the
                        qubit it measures.

                        Should return True if the test failed (you died.)


        Returns True if it is a live bomb, False otherwise, along with the
            post-acted-upon bombState.

        Side effects: None, but the tester might send you to jail or
            explode.

        >>> (a, _) = isItABomb(I, bombState=listToState(["0"]), qubit=1, tester=bombTester)
        >>> a
        False
        
        >>> (a, s) = isItABomb(X, bombState=listToState(["0"]), qubit=1, tester=bombTester)
        >>> all(s == listToState(["0"]))
        True
        >>> a
        True

        # If we reinterpret the results, we don't die on a "|1>" either.

        # >>> (a, _) = isItABomb(X, bombState=listToState(["1"]), qubit=1, didExplode=lambda x: x[0] == "0")
        # >>> a
        # True
    """
    if not N:
        # With N = 100, pr(success) from (3) of the paper is approximately 97.5%
        N = 100

    s0 = listToState(["0"])
    s1 = listToState(["1"])
    
    delta = np.pi/(2. * N)

    dimBomb = int(np.log2(len(bombState)))


    # R_delta on only the first qubit.
    Rdelta = np.kron( np.array([
        [ np.cos(delta), -np.sin(delta) ],
        [ np.sin(delta),  np.cos(delta) ] ]), np.identity(2**dimBomb) )

    # Set the bomb operator to act only on the qubit we've asked it to.
    bop = reduce(np.kron, [ np.identity(2**(qubit-1)),
        bombOperator,
        np.identity(2**(dimBomb-qubit)) ], 1)

    # Controlled-bomb operator
    C_bop = np.kron( np.dot(s0, ct(s0)), np.identity(2**dimBomb) ) + np.kron( np.dot(s1, ct(s1)), bop )

    # 1. Initial value.
    state = np.kron( listToState(["0"]), bombState )

    for k in xrange(N):
        # 1. Initialise second qubit to |0>. 
        #
        #   We choose to do this by measuring it in the computational basis,
        #   and if it is "|1>", apply "X" to it.

        # (a, state) = measure(state, basis="0,1", qubits=[2])
        # if a[0] == "1":
        #     state = np.dot( np.kron(I, X), state )

        
        # 2. Rotate the first qubit using R_delta.
        state = np.dot(Rdelta, state)
        
        # 3. Apply controlled-bomb
        state = np.dot(C_bop, state)

        # 4. Measure the qubit (ignoring the one we're rotating with) that
        # we've been asked to.

        (failed, state) = tester(state)

        assert not failed, "You have died."
    
    # So now, if the first register is |1>, we know that we were passed a dud.
    (a, state) = measure(state, basis="0,1", qubits=[1])

    # Give them back the state they gave us, now that we have acted on it. We
    # are going to hand them this state in the worst possible way; we're going
    # to use our knowledge of the control qubit to just take the relevant bit
    # of the resulting 'state' matrix. I.e. if 'a = 0', then take the first
    # half, and if 'a = 1', take the second half.
    
    if a[0] == "1":
        # Not a bomb.
        bomb = False
        finalState = state[len(state)/2:]
    else: # a[0] == "0"
        # Otherwise, live bomb!
        bomb = True
        finalState = state[0:len(state)/2]

    return (bomb, finalState)


def nsTester (s, state):
    """ We're going to ask the bank to measure the (relevant) bit of the state
        that we've been given.

        Returns (True, state) if the test failed, and
                (False, state) if it succeeded.
    """

    # HACK/XXX/TODO: We are actually politely asking the bank to only look at
    # the qubits that are relevant to the note. This is basically because our
    # states here aren't really QM states; when we slice up the arrays, we get
    # copies of them, and do not modify the original arrays (thankfully!).
    
    (valid, state) = validate((s, state), startingQubit=1)

    return (not valid, state)


def nsCounterfeit ( (s, keyState) ):
    """ Counterfeits according to the protocol of Daniel Nagaj and Or Sattath.

        (c.f. "naivelyCounterfeit")
    """

    # Let's cheat for a moment.
    (s, key) = generateMoneyData(20, 3)
    keyState = listToState(key)

    # We're continually modifying this.
    state = keyState

    guessedKey = []

    # Our plan: Forge Money.

    eps = 0.10
    N = int(np.pi**2 * n / (2. * eps))
    
    # import pdb
    # pdb.set_trace()
    print("key, ", key)

    def curriedNsTester (state):
        return nsTester(s, state)

    guesses = []
    for k in xrange(1, n+1):

        guess = None

        (isDud, state) = isItABomb( X, state, qubit=k, tester=curriedNsTester, N=N)

        if isDud:
            guess = "+"

        if not guess:
            (isDud, state) = isItABomb(-X, state, qubit=k, tester=curriedNsTester, N=N)
            if isDud: 
                guess = "-"

        if not guess:
            (a, state) = measure(state, basis="0,1", qubits=[k])
            guess = a[0]

        guesses.append(guess)

        assert all(state == keyState)

    print("guess ", guesses)
    # import pdb
    # pdb.set_trace()

    return ((s, listToState(guessedKey)), (s, state))

 
def naivelyCounterfeit ( (s, keyState) ):
    """
        Counterfeits the given (s, |k_s>) pair by measuring each qubit in the
        computational basis.

        >>> (s, forgedKeyState) = naivelyCounterfeit( getMoney(1000, seed=2) )
        >>> (a, _) = validate( (s, forgedKeyState) )
        >>> a
        False
    """
    
    # Our plan: Forge Money. We perform the forgery naively, by measuring
    # qubit in the computational basis. This gives is the right answer when
    # the state is in this basis, and the wrong answer 1/4 of the time
    # otherwise.

    guessedKey = []
    for i in xrange(1, n+1):
        (a, keyState) = measure(keyState, basis="0,1", qubits=[i])
        guessedKey.append( a[0] )

    return (s, listToState(guessedKey))


# Bank-related bits.
# -------------------------------

def validate ( (s, keyState), startingQubit=None ):
    """ Returns True if the money can be deposited. False otherwise, along
        with the state.

        (s, keyState):  The tuple returned by getMoney(amount).

        startingQubit:  (optional) The qubit at which to begin measuring the
                        keyState.

                        This is a hack.

        >>> (s, key) = generateMoneyData(20, 2)
        >>> (a, _) = validate( (s, listToState(key)) )
        >>> a
        True

        >>> (s1, key1) = generateMoneyData(20, 3)
        >>> (s2, key2) = generateMoneyData(20, 4 )
        >>> (a, _) = validate( (s2, listToState(key1)) ) # Mismatched key and serial number.
        >>> a
        False
        
        >>> (s1, key1) = generateMoneyData(20, seed=32)
        >>> (s2, key2) = generateMoneyData(20, seed=412)
        >>> (a, _) = validate( (s1, listToState(key2)) ) 
        >>> a
        False

        >>> (a, _) = validate( getMoney(50) )
        >>> a
        True
        """

    assert s in bankDatabase, "Key is not even in the database! What are you doing!"
    
    inState = copy.copy(keyState)

    if not startingQubit:
        startingQubit = 0

    # Alright, so our process here is to measure each bit in the basis that we
    # know it should be measured in.

    referenceKey      = bankDatabase[s]
    referenceKeyState = listToState(referenceKey)

    bases = { "0": "0,1", "1": "0,1", "+": "+,-", "-": "+,-" } 

    outcomes = []
    for (i, k) in enumerate(referenceKey):
        (a, keyState) = measure(keyState, basis=bases[k],
                qubits=[i+1+startingQubit])
        outcomes.append(a[0])

    # import pdb
    # pdb.set_trace()

    if outcomes != referenceKey:
        # Invalid key, send them to jail.
        return (False, keyState)
    
    return (True, keyState)


def generateMoneyData (amount, seed=None):
    """ Returns a tuple (s, k_s) where s is the serial number, and k_s is the
        key that will be used to build the money state.

        >>> random.seed(2)
        >>> np.random.seed(2)
        >>> (a, key) = generateMoneyData(20, seed=3)
        >>> a
        2
        >>> key[0:3]
        ['+', '1', '+']
        """

    if seed: random.seed(seed)

    s = random.randint(0, 2**n)

    # Let's generate a 7-bit key from the set of states {0,1,+,-}.
    alphabet = ["0", "1", "+", "-"]
    key = [ random.choice(alphabet) for k in xrange(n) ]

    # Save this key.
    bankDatabase[s] = key
    return (s, key)


def getMoney (amount, seed=None):
    """ Returns (s, |k_s>).
        """
    (s, key) = generateMoneyData(amount, seed)
    keyState = listToState(key)

    return (s, keyState)


if __name__ == "__main__":
    seed = 223

    random.seed(seed)
    np.random.seed(seed)

    # Note: argument is ignored for now.
    (s, forged) = nsCounterfeit( getMoney(100) )


# Some tests
# -------------------------------
def test_naive_counterfeiting ():

    iters = 100 # Let's have a few attempts at counterfeiting.
    outcomes = []
    for k in xrange(iters):
        (s, keyState) = naivelyCounterfeit( getMoney(50) )
        outcomes.append( validate((s, keyState)) )
    #
    trues  = len(filter(lambda x: x, outcomes))
    falses = len(outcomes) - trues

    assert trues < falses # Hopefully

    # We expect to succeed (3/4)^n percent of the time.

    prSuccess = 100 * (3/4.)**n
    
    assert falses/float(trues) < prSuccess
