import jax.numpy as jnp
import equinox as eq
from jx.nn import Linear
from equinox import nn as nnx
import jax
import optax


key = jax.random.key(10)
key_x, key_y, key_linear = jax.random.split(key, 3)
x = jax.random.normal(key_x, (10, 10))
y = jax.random.normal(key_y, (10, 10))

linear = Linear(
    10,
    10,
    bias = False,
    key = key_linear
)


@eq.filter_value_and_grad
def loss_fn(model, x, y):
    y_hat = jax.vmap(model)(x)
    loss = jnp.mean((y -y_hat)**2)
    return loss 

eq.filter_jit
def step(model, x , y, opt_state):
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eq.apply_updates(model, updates)


    return loss, model, opt_state



optim = optax.adam(1e-3)
opt_state = optim.init(eq.filter(linear, eq.is_array))


for i in range(10000):
    loss, linear, opt_state = step(linear, x, y, opt_state)
    print(f"DEBUGPRINT[100]: test_grad.py:44: loss={loss}")

