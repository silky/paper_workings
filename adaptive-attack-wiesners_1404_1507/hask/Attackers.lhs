> {-# LANGUAGE GADTs                  #-}
> {-# LANGUAGE FlexibleInstances      #-}
> {-# LANGUAGE FunctionalDependencies #-}
> {-# LANGUAGE MultiParamTypeClasses  #-}
> {-# LANGUAGE TemplateHaskell        #-}
> {-# LANGUAGE TypeSynonymInstances   #-}
>
> module Attackers (
>     
>  ) where
>
> import           Control.Lens        ( makeFields
>                                      , (^.) -- View
>                                      , (.~) -- Set
>                                      , (&)  -- Combinator
>                                      )
> import           System.Cmd          ( system
>                                      )
> import           System.Random       ( randomR
>                                      , randomRs
>                                      , StdGen
>                                      )
> import           Control.Monad.State ( State
>                                      , evalState
>                                      , put
>                                      , get
>                                      )
> import qualified Data.Map as M
> import           Utils               ( indexes
>                                      , basisToState
>                                      , mdim
>                                      , measure
>                                      , basisVectors
>                                      , operatorAt
>                                      )
> import           Data.Packed.Matrix  ( Matrix
>                                      , mapMatrix
>                                      )
> import           Numeric.LinearAlgebra.HMatrix -- (kronecker, dot)
> import           Numeric.LinearAlgebra.Data
> import           System.Random
> import           Numeric.LinearAlgebra (ctrans)
> import           Defs 
> import           Banks 

> gen :: StdGen
> gen = mkStdGen 0

Let's define an insane infix operator.


> -- In vim this is "Ctrl-K +_".
> (〄) :: (Product t) => Matrix t -> Matrix t -> Matrix t
> (〄) a b = kronecker a b

The naive counterfeiting approach is, say, measure every qubit in the standard
basis and hope that when the qubit wasn't constructed in the standard basis
that it just "works". Of course, it will fail with 50% probability in those
circumstances.


> naivelyCounterfeit :: Money -> Money
> naivelyCounterfeit money = newMoney
>     where
>       keySize = mdim (money ^. keyState)
>       (guessedKey, newState) = foldl f ([], money ^. keyState) [1..keySize]
>           where f (xs, state') i = (bs ++ xs, state'')
>                    -- | Note we fix the basis here; we can't do anything
>                    -- else. Or can we!?!?!? (Spoiler: Yes.)
>                    where (state'', bs) = measure gen state' [i] (basisVectors "0")
>       newMoney = money & keyState .~ newState


We're going to need the "isBomb" function. This does the interesting work.
Actually this also needs to perform arbitrary IO, as we could be exploded when
performing this test.


> isBomb :: Operator
>        -> QuantumState
>        -> Int
>        -> (QuantumState -> ValidateResponse)
>        -> (Bool, QuantumState)
> isBomb op bombState qubit tester = result
>   where
>       _N = 100
>       --
>       -- Why types. WHY?!?!
>       delta   = pi / (2 * (fromIntegral _N)) :: Complex Double
>       dimBomb = mdim bombState
>       rDelta  = (2><2) [ cos delta, -(sin delta), sin delta, cos delta ]
>                   〄 ident (2 ^ dimBomb)
>       bop     = operatorAt op qubit dimBomb
>       cBop    = mul s0 (ctrans s0) 〄 ident 2 ^ dimBomb +
>                   mul s1 (ctrans s1) 〄 bop
>       --
>       state   = foldl f (s0 〄 bombState) [1.._N]
>           -- This seems INSANELY bad.
>           where f state' i = (flip evalState) state' (do
>                   s <- get
>                   put $ mul rDelta s
>                   s <- get
>                   put $ mul cBop   s
>                   s <- get
>                   case (tester s) of
>                       Left m  -> return $ m ^. keyState
>                       -- TODO: Somehow evaluate the IO !?
>                       Right _ -> error "You died."
>                   )
>       (newState, outcomes) = measure gen state [1] (basisVectors "0")
>       sadness = toList (flatten newState)
>       sadlen  = length sadness
>       detectedBomb = head outcomes == "1"
>       much = if detectedBomb then
>                           take (sadlen `div` 2) sadness
>                           else
>                           drop (sadlen `div` 2) sadness
>       result = (detectedBomb, (dimBomb><1) much)
>
>
>       -- result = (head outcomes == "1", discard [1] newState)



This is a tool used by the Nagaj-Sattath counterfeiting scheme. I'm not sure
what value it is providing. I don't think it's even needed.

Perform the Nagaj-Sattath counterfeiting scheme.

> nagajSattathCounterfeit :: (Bank b) => b -> Money -> Money
> nagajSattathCounterfeit bank money = newMoney
>     where
>       keySize = mdim (money ^. keyState)
>       (guessedKey, _) = foldl f ([], money ^. keyState) [1..keySize]

Here we go about our guessing procedure. We use the bomb detection device to
check if a few operations are bombs.

>           where f (xs, state') i =
>                   let (g, state'') = (guess bank money) state' i
>                   in (g : xs, state'')
>       newMoney = money & keyState .~ (basisToState guessedKey)


This is how we guess.

Note: This code makes me very sad. It shouldn't be, but I don't know how to
make it do.

> guess :: (Bank b)
>       => b
>       -> Money
>       -> QuantumState
>       -> Int
>       -> (BasisId, QuantumState)
> guess b money inState i = evalState (do
>   (_, state) <- get
>   let validateState s' = validate b (money & keyState .~ s')
>   let (bomb, state) = isBomb mX state i validateState 
>   if not bomb
>       then put (Just "+", state)
>       else do
>           let (bomb, state) = isBomb (mapMatrix negate mX) state i validateState
>           if not bomb then put (Just "-", state)
>               else do
>                   let (state, bs) = measure gen state [i] (basisVectors "0")
>                   put (Just ".", state)
>   (Just x, state) <- get
>   return (x, state)) (Nothing, inState)


