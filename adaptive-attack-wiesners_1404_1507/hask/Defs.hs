{-# LANGUAGE GADTs                  #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE TemplateHaskell        #-}
{-# LANGUAGE TypeSynonymInstances   #-}

module Defs
  where

import Control.Lens                  (makeFields)
import Data.Complex                  (Complex)
import Numeric.LinearAlgebra.HMatrix
import Numeric.LinearAlgebra.Data

type CurrencyUnits = Int
type BasisId       = String
type SerialNumber  = Int
type Key           = [BasisId]
-- type QuantumState  = Vector (Complex Double)
type QuantumState  = Matrix (Complex Double)
type Operator      = Matrix (Complex Double)-- TODO: Should be a matrix.

-- Need some notion of complex numbers.

data Money = Money {
      _mAmount       :: CurrencyUnits
    , _mSerialNumber :: SerialNumber
    , _mKeyState     :: QuantumState
    }
$(makeFields ''Money)


mX :: Operator
mX = (2><2) [0, 1, 1, 0]


-- | Some standard states.
s0 :: Operator
s0 = (2><1) [1, 0]

s1 :: Operator
s1 = (2><1) [0, 1]

sp :: Operator
sp = (1/ sqrt 2.0) * (2><1) [1, 1]

sm :: Operator
sm = (1/ sqrt 2.0) * (2><1) [1, 1]
