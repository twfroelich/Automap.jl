cd(@__DIR__)
using Pkg
Pkg.activate(".");
Pkg.instantiate()

using Automap, Flux, Metalhead, DataSets, CUDA, Optimisers, HDF5, FFTW, Images

function automap(patch_size,dropout)
  m = Chain(  
      Flux.Dense((patch_size).^2*2, (patch_size).^2, tanh),
      Flux.Dropout(dropout),
      Flux.Dense((patch_size).^2, (patch_size).^2, tanh),
      Flux.Dropout(dropout),

      x -> reshape(x, (patch_size,patch_size,1,:)),
  
      Flux.Conv((5,5), 1 => patch_size,    relu; stride = 1, pad = 2),
      Flux.Dropout(dropout),
      Flux.Conv((5,5), patch_size => patch_size,   relu; stride = 1, pad = 2),
      Flux.Dropout(dropout),
      Flux.ConvTranspose((7,7), patch_size => 1; stride = 1, pad = 3),

      Flux.flatten,
  )
  return m
end

## ---
model = automap(64,0.004;)

key = open(BlobTree, DataSets.dataset("imagenet_cyclops")) do data_tree
         Automap.train_solutions(data_tree, path"train_data_key.csv")
       end;

val = open(BlobTree, DataSets.dataset("imagenet_cyclops")) do data_tree
         Automap.train_solutions(data_tree, path"val_data_key.csv")
       end;
## ---

opt = Optimisers.RMSProp(2e-6,0.6);

setup, buffer = prepare_training(model, key,
                                        CUDA.devices(),
                                        opt, # optimizer
                                        96,  # batchsize per GPU
                                        epochs = 2);

#loss = Flux.Losses.logitcrossentropy
loss = Flux.Losses.mse

out = AutomapNet.train(loss, setup, buffer, opt,
                            val = val,
                            sched = identity);
