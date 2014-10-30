> {-# LANGUAGE GADTs                  #-}
> {-# LANGUAGE FlexibleInstances      #-}
> {-# LANGUAGE FunctionalDependencies #-}
> {-# LANGUAGE MultiParamTypeClasses  #-}
> {-# LANGUAGE TemplateHaskell        #-}
> {-# LANGUAGE TypeSynonymInstances   #-}
>
> module Banks (
>       StandardBank
>     , validate
>     , ValidateResponse
>     , Bank
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
>                                      , put
>                                      , get
>                                      )
> import qualified Data.Map as M
> import           Utils               ( indexes
>                                      , basisToState
>                                      , measure
>                                      , basisVectors
>                                      )
> import           Defs 

Bank implementations
==

Here we define functions for withdrawing money from banks, and implement the
banks validation procedures.

- - -

Types and classes
==

We begin with some type alises. We need a "Database" of keys indexed by serial
number, we need a "State" for the bank, in which we are allowed to add new
keys to the database, and we also need to define the "response" that the bank
provides when we've asked it to the validate the money.

In this case, we allow the bank to do arbitrary IO things; i.e. perhaps it
will send us to jail, or shoot us, or something.

> -- | The thing that maps money serial numbers to the key that was used to
> -- generate them.
> type Database  = M.Map SerialNumber Key

> -- | The "world" that the bank operations will be performed in. The
> -- BankFacilities all hold a database. Note that this is looking for
> -- any extra parameter still; which will typically the be "BankFacilities"
> -- that we modify.
> type BankState = State BankFacilities

> -- | Either we give you your money back, or we explode.
> type ValidateResponse = Either Money (IO ())


A *Bank* will be a thing that we can withdraw money from, and a thing which
will tell us the legitimacy of money that we have.

> class Bank b where
>    -- | Wherein the bank takes an amount of money and decides whether or not
>    -- it represents actual or forged currency.
>    validate :: b -> Money -> ValidateResponse
>
>    -- | Obtain some money from the bank. This will result in a state change,
>    -- as the bank needs to record this transaction.
>    withdraw :: b -> CurrencyUnits -> BankState Money


> data BankFacilities = BankFacilities {
>      _bfDatabase  :: Database
>
>    -- | The random number generator.
>    , _bfGenerator :: StdGen
>    }
> $(makeFields ''BankFacilities)


> -- | A bank that generates money in the standard fashion.
> data StandardBank  = StandardBank {
>    _sbFacilities :: BankFacilities
>    }
> $(makeFields ''StandardBank)

- - -

Some constants
==

The key size that we'll be working with. We want this to be small, because the
size of the state vector is $2^\text{keySize}$.

> keySize :: Int
> keySize = 5


The bad thing we're doing to do to people that try to cross us.

> badThing :: IO ()
> badThing = do
>   system "animate explosion.gif"
>   return ()


Standard Bank
==

Let us now define the features of the **StandardBank**.

> instance Bank StandardBank where

The *StandardBank* generates money in the typical way. It will

  1. Pick an alphabet $\mathcal{A}$ containing basis elements, here we will
  let $\mathcal{A} = \{ |0\rangle, |1\rangle, |+\rangle, |-\rangle \}$.

  2. Pick a random subset, $k$, (of size $n$ - the key size) to form our key.

  3. Pick a random serial number, and associate this serial number to this key
  in the database.

  4. Return a quantum state built from this key (i.e. a piece of "Money".)

> -- | Generates "normal" quantum money.
>    withdraw b amount = do
>      bf <- get
>      let (serial, gen) = randomR (1, 2 ^ keySize) (bf ^. generator)
>
>      let alphabet = ["0", "1", "+", "-"] :: [BasisId]
>          indicies = take keySize (randomRs (1, length alphabet) gen)
>          key      = indexes indicies alphabet
>          keyState = basisToState key
>
>      let newBf = BankFacilities newDb gen
>          newDb = M.insert serial key (bf ^. database)
>
>      let money = Money amount serial keyState
>      put newBf
>      return money


Now for validation. The process is, for each qubit, measure it in the basis
that we know it was constructed from. In this way, reconstruct a candidate
key. If the key we have on file doesn't match this candidate key, then kill
them.

>    validate b money = result
>      where
>        actualKey = (b ^. facilities ^. database) M.! (money ^. serialNumber)
>        gen       = (b ^. facilities ^. generator)

We will calculate the *derivedKey* by the measurement procedure. The actual
key tells us what basis we should measure in. For example, we may have

$$ \begin{aligned}
  \text{actualKey} = \{ ``0", ``1", ``+", ``+" \}
\end{aligned} $$

and so the bases we will measure the incoming key state,
    $|\psi\rangle =$ `money ^. keyState`, in is

$$ \begin{aligned}
  \text{basis} = \{ ``0/1", ``0/1", ``+/-", ``+/-" \}
\end{aligned} $$

At the moment the below doesn't handle measuring multiple qubits; but infact
the standard bank doesn't need to do that (only the EntangledBank).

>        -- | "derived" here contains the final measured state, and list of
>        -- measurement outcomes.
>        (derivedKey, newState) = foldl f ([], money ^. keyState) (zip actualKey [1..])
>          where f (xs, state') (basisId, i) = (bs ++ xs, state'')
>                       where (state'', bs) = measure gen state' [i] (basisVectors basisId)
>        newMoney = money & keyState .~ newState
>        result = if derivedKey == actualKey then Left newMoney else Right badThing

