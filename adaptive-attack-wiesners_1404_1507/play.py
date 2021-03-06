#!/usr/bin/env python
# coding: utf-8

""" Plays around with Wiesner's quantum money scheme and this papers attack on
    it - https://scirate.com/arxiv/1404.1507

    All of this is in qubits.

    Note that none of this has been written to be very efficient; it could be
    dramatically improved, if one wished.
    """


# WARNING: Numpy and random use different seeds.
import random
import copy
from memoize import memoize
import numpy as np

__version__ = "0.1"

I = np.identity(2)
X = np.array([[0,1],[1,0]])
H = np.array([[1,1],[1,-1]])/np.sqrt(2)
HH = np.kron(H, H)

states = {
    "0": np.array([[1], [0]]),
    "1": np.array([[0], [1]]),
    "+": np.array([[1], [1]])/np.sqrt(2),
    "-": np.array([[1], [-1]])/np.sqrt(2),
    "r3+": np.array([[1], [np.sqrt(2)]])/np.sqrt(3),
    "r3-": np.array([[np.sqrt(2)], [-1]])/np.sqrt(3),
}

# Add in the bell-basis states.
states["b1"] = (np.kron(states["0"], states["1"]) + np.kron(states["1"],
        states["0"]))/np.sqrt(2)
states["b2"] = (np.kron(states["0"], states["1"]) - np.kron(states["1"],
        states["0"]))/np.sqrt(2)
states["b3"] = (np.kron(states["0"], states["0"]) + np.kron(states["1"],
        states["1"]))/np.sqrt(2)
states["b4"] = (np.kron(states["0"], states["0"]) - np.kron(states["1"],
        states["1"]))/np.sqrt(2)

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

def setseed (seed):
    random.seed(seed)
    np.random.seed(seed)



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
    """
    if not N:
        # With N = 100, pr(success) from (3) of the paper is approximately 97.5%
        N = 100

    s0 = listToState(["0"])
    s1 = listToState(["1"])
    
    delta = np.pi/(2. * N)

    dimBomb = dim(bombState)

    # R_delta on only the first qubit.
    Rdelta = np.kron( np.array([
        [ np.cos(delta), -np.sin(delta) ],
        [ np.sin(delta),  np.cos(delta) ] ]), np.identity(2**dimBomb) )

    # Set the bomb operator to act only on the qubit we've asked it to.
    bop = operatorAt(bombOperator, qubit, dimBomb)

    # Controlled-bomb operator
    C_bop = np.kron( np.dot(s0, ct(s0)), np.identity(2**dimBomb) ) + np.kron( np.dot(s1, ct(s1)), bop )

    # 0. Initial value.
    state = np.kron( listToState(["0"]), bombState )

    for k in xrange(N):
        # 1. Initialise second qubit to |0>. 
        #
        #    (Actually, this step isn't necessary. From figure (1) we'll
        #    always be in the state we need after measurement (otherwise we'd
        #    die) and similarly for the bank protocol.

        
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


def nsTester (s, inState):
    """ We're going to ask the bank to measure the (relevant) bit of the state
        that we've been given.

        Returns (True, state) if the test failed, and
                (False, state) if it succeeded.
    """
    state = copy.copy(inState)

    # HACK/XXX/TODO: We are actually politely asking the bank to only look at
    # the qubits that are relevant to the note. This is basically because our
    # states here aren't really QM states; when we slice up the arrays, we get
    # copies of them, and do not modify the original arrays (thankfully!).
    
    (valid, state) = validate((s, state), startingQubit=1)

    # if not valid:
    #     import pdb
    #     pdb.set_trace()

    return (not valid, state)


def nsCounterfeit ( (s, keyState) ):
    """ Counterfeits according to the protocol of Daniel Nagaj and Or Sattath.

        (c.f. "naivelyCounterfeit")

        Returns two tuples of (s, |k_s>), where the first tuple is the forged
        one, and the second one is the key we derived it from.

        >>> setseed(6)
        >>> (forged, original) = nsCounterfeit(getMoney(100, n=3))
        >>> (a, _) = validate(forged)
        >>> a
        True
        >>> (a, _) = validate(original)
        >>> a
        True
    """

    n = dim(keyState)

    # We're continually modifying this.
    state = keyState

    # Our plan: Forge Money.

    eps = 0.10
    N = int(np.pi**2 * n / (2. * eps))
    
    def curriedNsTester (state):
        return nsTester(s, state)

    guessedKey = []
    for k in xrange(1, n+1):

        guess = None

        # Following section 3, we check if the bomb tester detects "duds" for
        # "X" and "-X". If not, it's in the computational basis, so we measure
        # it, and we're done.

        (bomb, state) = isItABomb(X, state, qubit=k, tester=curriedNsTester, N=N)

        if not bomb:
            guess = "+"

        if not guess:
            (bomb, state) = isItABomb(-X, state, qubit=k, tester=curriedNsTester, N=N)
            if not bomb: 
                guess = "-"

        if not guess:
            # This operation breaks if the key is entangled.
            (a, state) = measure(state, basis="0,1", qubits=[k])
            guess = a[0]

        assert all(abs(state - keyState) < 1e-10), "We broke the original."

        guessedKey.append(guess)

    guessedKeyState = listToState(guessedKey)

    return ((s, guessedKeyState), (s, state))

 
def naivelyCounterfeit ( (s, keyState) ):
    """
        Counterfeits the given (s, |k_s>) pair by measuring each qubit in the
        computational basis.

        >>> setseed(3)
        >>> (s, key) = generateMoneyData(1000, n=3)
        >>> assert '+' in key
        >>> (forged, original) = naivelyCounterfeit( (s, listToState(key)) )
        >>> (a, _) = validate(forged)
        >>> a
        False

        (We expect a fail above because the key is not entirely in the
        computational basis.)
    """
    
    n = dim(keyState)

    # Our plan: Forge Money. We perform the forgery naively, by measuring
    # qubit in the computational basis. This gives is the right answer when
    # the state is in this basis, and the wrong answer 1/4 of the time
    # otherwise.

    guessedKey = []
    for i in xrange(1, n+1):
        (a, keyState) = measure(keyState, basis="0,1", qubits=[i])
        guessedKey.append( a[0] )

    return ((s, listToState(guessedKey)), (s, keyState))


# Bank-related bits.
# -------------------------------

def validate ( (s, keyState), startingQubit=None ):
    """ Returns True if the money can be deposited. False otherwise, along
        with the state.

        (s, keyState):  The tuple returned by getMoney(amount).

        startingQubit:  (optional) The qubit at which to begin measuring the
                        keyState.

                        This is a hack.

        >>> setseed(2)
        >>> (s, key) = generateMoneyData(20, n=3)
        >>> (a, _) = validate( (s, listToState(key)) )
        >>> a
        True

        >>> setseed(3)
        >>> (s1, key1) = generateMoneyData(20, n=3)
        >>> (s2, key2) = generateMoneyData(20, n=3)
        >>> (a, _) = validate( (s2, listToState(key1)) ) # Mismatched key and serial number.
        >>> a
        False
        
        >>> setseed(32)
        >>> (s1, key1) = generateMoneyData(20, n=3)
        >>> (s2, key2) = generateMoneyData(20, n=3)
        >>> (a, _) = validate( (s1, listToState(key2)) ) 
        >>> a
        False

        >>> (a, _) = validate( getMoney(50, n=3) )
        >>> a
        True

        >>> setseed(3)
        >>> (s, eKey) = generateEntangledMoney(10, n=5)
        >>> eKey
        ['1', '+', '+', 'b1']
        >>> (a, after)    = validate( (s, listToState(eKey)) )
        >>> a
        True
        >>> all((after - listToState(eKey) < 1e-10).tolist())
        True
        """

    assert s in bankDatabase, "Key is not even in the database! What are you doing!?"
    
    inState = copy.copy(keyState)

    if not startingQubit:
        startingQubit = 0

    # Alright, so our process here is to measure each bit in the basis that we
    # know it should be measured in.

    referenceKey = bankDatabase[s]

    bases = { "0": "0,1", "1": "0,1", "+": "+,-", "-": "+,-",
            "b1": "Bell", "b2": "Bell", "b3": "Bell", "b4": "Bell",
            "r3+": "r3", "r3-": "r3"
            } 

    outcomes = []
    shift    = 0 
    for (i, k) in enumerate(referenceKey):
        qubitsToMeasure = [i+1+startingQubit+shift]

        (a, keyState) = measure(keyState, basis=bases[k], qubits=qubitsToMeasure)
        outcomes.append(a[0])

        if "b" in k: # Insane way of encoding this.
            # HACK/XXX/TODO
            #
            # Because we're going to measure the next qubit as well, if this
            # qubit was in the Bell basis, we need to shift the index of the
            # subsequent measurements.
            shift = shift + 1


    if outcomes != referenceKey:
        # Invalid key, send them to jail.
        return (False, keyState)
    
    return (True, keyState)


def generateEntangledMoney (amount, n):
    # Serial number.
    s = random.randint(0, 2**n)

    twos = random.randint(0, int(n/2.))
    ones = n - 2*twos

    twoQubit = [ "b1", "b2", "b3", "b4" ]
    oneQubit = ["0", "1", "+", "-" ]

    key =  [ random.choice(oneQubit) for k in xrange(ones) ]
    key += [ random.choice(twoQubit) for k in xrange(twos) ]

    # Save this key.
    bankDatabase[s] = key
    return (s, key)


def generateSomeOtherMoney (amount, n):
    s = random.randint(0, 2**n)

    # Let's generate a n-bit key from the set of states {0,1,+,-}.
    alphabet = ["0", "1", "r3+", "r3-"]
    key = [ random.choice(alphabet) for k in xrange(n) ]

    # Save this key.
    bankDatabase[s] = key
    return (s, key)



def generateRegularMoney (amount, n):
    s = random.randint(0, 2**n)

    # Let's generate a n-bit key from the set of states {0,1,+,-}.
    alphabet = ["0", "1", "+", "-"]
    key = [ random.choice(alphabet) for k in xrange(n) ]

    # Save this key.
    bankDatabase[s] = key
    return (s, key)


def generateMoneyData (amount, n):
    """ Returns a tuple (s, k_s) where s is the serial number, and k_s is the
        key that will be used to build the money state.
        """
    return generateRegularMoney(amount, n)


def getMoney (amount, n):
    """ Returns (s, |k_s>).
        """
    (s, key) = generateMoneyData(amount, n)
    keyState = listToState(key)

    return (s, keyState)


if __name__ == "__main__":
    n = 4 # Say.

    # (s, key) = generateEntangledMoney(1000, n)
    print("n ", n)
    (s, key) = generateSomeOtherMoney(1000, n)

    # key = ["0", "1", "b4"]
    # key = ["0", "1", "b3"]

    # key = ["b3"]
    bankDatabase[s] = key

    print("Planning on counterfeiting key: |{0}>, #{1}.".format("".join(key), s))
    (forged, original) = nsCounterfeit( (s, listToState(key)) )

    (a, _) = validate(forged)
    (b, _) = validate(original)

    if a and b:
        print("Success! We forged a {0:d}-qubit key!".format(n))
    else:
        print("We went to jail.")



# Some tests
# -------------------------------
def test_naive_counterfeiting ():

    iters = 100 # Let's have a few attempts at counterfeiting.
    outcomes = []
    for k in xrange(iters):
        (s, key) = generateMoneyData(5)
        (forged, original) = naivelyCounterfeit( (s, listToState(key)) )
        (valid, _) = validate(forged)
        outcomes.append(valid)
    #
    trues  = len(filter(lambda x: x, outcomes))
    falses = len(outcomes) - trues

    assert trues < falses # Hopefully

    # We expect to succeed (3/4)^n percent of the time.

    prSuccess = 100 * (3/4.)**n
    
    assert falses/float(trues) < prSuccess
