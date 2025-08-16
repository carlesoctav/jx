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
