import jax
import equinox as eqx 
from equinox import nn


class Model(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear


    def __init__(
        self,
    ):
        key = jax.random.key(0)
        key1, key2 = jax.random.split(key)
        self.linear1 = nn.Linear(10, 10, key=key1)
        self.linear2 = nn.Linear(10, 10, key=key2)

def init(self,):
    pass

def main():

    key = jax.random.key(10)
    model = Model()
    a, b = eqx.partition(model, eqx.is_array)
    print(f"DEBUGPRINT[52]: test_hallo.py:24: b={b}")
    print(f"DEBUGPRINT[51]: test_hallo.py:24: a={a}")
    model.linear1.__call__ = init
    # model.linear1.weight = jax.random.normal(key, (10,10))
    print(model.linear1.weight)


if __name__ == "__main__":
    main()

