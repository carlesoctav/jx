pretty much i want a Trainer(
)

Trainer(model_config, dataset_config, optimizer_config)

fabric.setup()

parallel_config = dp, fsdp, tp, etc

model = shard(model, *, parallel_config = parallel_config, manually_sharding: regex_pattern->annotation) #apply sharding annotation into the model

model = distributed(config, parallel_config)

model offloading direclty into the gpu

-> should be compatible with other classmethod that create that model class
model = shard(ModelClass, *args) make a class directly inside the jitted function

modle = shard(ModelClass,



@shard_map
in_pspecs
outpspecs
train(model, batch)-> model:
    pass


so let say we have Linear
then we create FSDPLInear by nn.map_variables() since we're not adding map_out_fn
how can i capture the sharded state for in_specs and out_specs fo jax.shard_map()


we chould simply create a function called get_parition_spec_fsdp that run/imitate this "sharding process"

but i want this to be composable with other things, like what if we also include TP, we map.variabels(FSDPLinear) on another axis, and try to support TP
we need to get the same specs and out spec again for this and potentially combining this two things together.
(also i think we should make TPLinear first before FSDP, so it would be FSDPTPLinear()) 

if we've map_out_fn but without outputing the modified pytree, 

FSDPTPLinear
    map_in FSDP (g
    map_in TP (just all gather for column wise)
    map_out TP (sharded back this and we can check the partition_spec)
    map_out FSDP (use the shard info from the map_out TP, and collective capture a correct sharding annoatation)

by this i think map_out is needed, because we can capture the state easily by just run model_state = jax.eval_shape(one_pass, model, x) where one_pass is function to run the model once
and then run get_parition_spec(model_state)

the problem that we need shard_map to run the one_pass because of the map_in fsdp and map_in TP required jax.lax.all_gather()
but i think we can create a function that just remove the map_in and just simulate the map_out TP and map_out FSDP hence we get the last state specs for inspecs and out specs

FSDPLinear

map_variables(
    target,
    map_in_fn -> just modifeid the view of array (like all_gather, masking ,etc)
    state_map_out_fn -> required to capture the in_specs, and out_specs (mostly both are the same thing), but doesnt add to the __call__
)

the idea is to create out_spec() method the output are the state when we run state_map_out_fn
on a compose like FSDPTPLinear, we'll replace the old out_spec() with warpped
somehting like this
    @ft.wraps(prev_out_spec)
    wrapped():
        state = prev_out_spec(*args, **kwargs) (state should be jax.eval_shaped, and have the array.valueis ararystructshape (see Darray))
        state_map_out_fn(state)

    we check if we've prev_out_sepc by checking it subclass to PartitionSpecLayerMixin

we want to get their partitionspec
    
get_partition_spec(model: eq.Module):
    state = model.get_partition_spec()

another problem, what if we've nested module here
SuperLinear:
    Linear1:
    linear2:

since we want to use this map_variables on type/class, instead of object we need this iter_module_type(module_type: type[eq.Module]) see how (nnx.Module does this, but remember it' iterate on object instead of type)

the idea is we create newtype one by one,
and we use dataclasses.fields(module_type) to extract all the field that suppose to be eq.module 

field = []
field_name, type_x in iter_type_module(model_type):
    new_type_x = fully_shard(typex) or tp_shard(type_x) or full_shard(tp_shard(type_x))
    field.append([field_name, type_x])



make_new_module(model_type, field)


the problem is what if we've nested like
SuperSUperLInear:
ou    superlinear1:
        linear1
        linear2:
    superlinear2:
        linear1
        linear2:


but we want only to shard the linear instead of superlinear
here's some idea, iter_type_module should recursiveley visit the nested until the smallest module
(so we need a registry to store the atomic value of module? idk check how nnx iter_module works)

field_aware_nested: idk what type is it
field_name, type_x in iter_type_module(model_type):
    new_type_x = fully_shard(typex) or tp_shard(type_x) or full_shard(tp_shard(type_x))
    field_aware_nested([field_name, type_x])


make_module(model, field_aware_nested)

you've flax_research and equinox_research to do research for this pattern
