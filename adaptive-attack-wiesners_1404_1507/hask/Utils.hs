module Utils ( indexes
             , basisToState
             , measure
             , discard
             , basisVectors
             , mdim
             , operatorAt
             )
  where

import           Defs
import qualified Control.Monad.Random          as R
import           System.Random                 ( mkStdGen
                                               , StdGen)
import           Numeric.LinearAlgebra.HMatrix
import           Numeric.LinearAlgebra         ( ctrans)
import           Numeric.LinearAlgebra.Data    ( ident
                                               , rows)
import           Data.Packed.Vector            ( dim)


-- | Creates a large operator with the single-qubit operator acting on the
-- given qubit.
operatorAt :: Operator -> Int -> Int -> Operator
operatorAt op pos dim = newOp
      where
          opDim = mdim op
          xs    = [ident 2 ^ (pos - 1), op, ident $ 2 ^ (dim-(pos - 1 + opDim))]
          newOp = foldl kronecker (ident 1) xs


-- | Calculates the dimension of this matrix, just by looking at the number of
-- rows. All our matrices are square.
mdim :: Operator -> Int
mdim o = truncate $ logBase 2 (fromIntegral (rows o))


-- | Calculates the number of qubits making up this state.
-- qdim :: QuantumState -> Int
-- qdim q = truncate $ logBase 2 (fromIntegral (dim q))


-- | Given a list of indicies, and a list, return those elements
-- of the array for those indicies.
indexes :: [Int] -> [a] -> [a]
indexes ixs xs = map (\i -> xs !! i) ixs


-- | Given a list of Basis Identifiers, return an actual state vector.
basisToState :: [BasisId] -> QuantumState
basisToState xs = foldl (kronecker) (ident 1) (map (\x -> case x of
                      "0" -> s0
                      "1" -> s1
                      "+" -> sp
                      "-" -> sm
                      ) xs)


-- | Measures a given state, returin a new (measured) state, and a list of
-- basis identifiers for the resulting eigenvalue that comes from the
-- measurement.
measure :: StdGen
        -> QuantumState 
        -> [Int] 
        -> [(BasisId, QuantumState)]
        -> (QuantumState, [BasisId])
measure gen state qubits basis = result
  where
      n      = mdim state
      result = foldl f (state, []) [1..n]
      f (state', xs) i = undefined -- (mState, label : xs)
        where
            (label,  mState) = R.runRand (R.fromList probs) gen
            -- "probs" is now a list
            probs = map g basis
            g (label, ei)  = ((label, mState), prob)
              where proj   = (operatorAt (mul ei (ctrans ei)) i n) 
                    mState = mul proj state
                    -- Stupid 0-indexing.
                    prob :: Rational
                    prob  = toRational (realPart $ atIndex (mul (ctrans mState) mState) (0,0))


-- | Discards the particular qubits from the given state.
discard :: [Int] -> QuantumState -> QuantumState
discard = undefined


-- | Given a particular basis identifier, return a list of basis states.
basisVectors :: BasisId -> [(BasisId, QuantumState)]
basisVectors b =
    let standardBasis = [("0", s0), ("1", s1)]
        hadamardBasis = [("+", sp), ("-", sm)]
     in case b of
         "0" -> standardBasis
         "1" -> standardBasis
         "+" -> hadamardBasis
         "-" -> hadamardBasis
