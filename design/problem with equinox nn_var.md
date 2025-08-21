see with mutable = True the outputs are tuple of (output, model)

then when we create a model we need to consider this scenario

class Model:
    linear: FSDPLinear (map.variable(Linear, map_in_fn, map_out_fn, mutable = True))
    linear2: FSDPLinear (map.variable(Linear, map_in_fn, map_out_fn, mutable = True))



    def __call__(x):
        params, static = eq.partition(self)
        out, modified_linear = linear(x)
        out, modified_linear = linear2(out)
        return out, eq.combine(modified_linear1, modified_linear2, static)


which is pain, compare to the flax imlementation of map_variable (read flax linen map_variables)
how can i achieve this on equinox
should we replicate the lift_transforms, and lift.map_variables functionality
i dont want to care about the tuple output even tho i set mutable = True

