# Automap.jl

This work is based off of ResNetImageNet.jl created by Dhairya L Gandhi. For the original project, see https://github.com/DhairyaLGandhi/ResNetImageNet.jl

### To start:

Start Julia with the environment of the package activated. This is currently necessary. Start julia with more threads than available. Finally, set up the environment via `] instantiate`.

There is a `Data.toml` in the repo which has a few example Datasets. The package uses the `"automap_cyclops"` data set by default. Make sure to update the path to where in the system the a dataset is available.

```julia
julia> using Automap, Flux, Metalhead, DataSets, CUDA, Optimisers, HDF5, FFTW, Images

julia> function Automap(patch_size,dropout)
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

julia> model = Automap(64,0.004);

julia> key = open(BlobTree, DataSets.dataset("automap_cyclops")) do data_tree
                Automap.train_solutions(data_tree, path"train_data_key.csv")
              end;

julia> val = open(BlobTree, DataSets.dataset("automap_cyclops")) do data_tree
                Automap.train_solutions(data_tree, path"val_data_key.csv")
              end;

julia> opt = Optimisers.RMSProp(2e-6,0.6)
Optimisers.RMSProp{Float64}(2.0e-6, 0.6, 2.220446049250313e-16)

julia> setup, buffer = prepare_training(model, key,
                                        CUDA.devices(),
                                        opt, # optimizer
                                        3,  # batchsize per GPU
                                        epochs = 2);

julia> loss = Flux.Losses.mse
mse (generic function with 1 method)

julia> Automap.train(loss, setup, buffer, opt,
                            val = val,
                            sched = identity);
```

Here `model` is specifically referring to Automap but can be subsituted for others, `key` describes a table of data and how it may be accessed. This table was generated for the NYU FASTMRI single knee dataset. See `test_fun.jl` for the functions used to generate these `key`s. It is important to note that the structure of the dataset is such that each `.h5` file contains an random number of complex images (i.e. file_001.h5 may contain 200 images and file_012.h5 may contain 10). The code iterates though each image in a given data file and resizes them to 64x64 to ensure they fit in memory. The resulting data will look somehting like file_001.h5 -> 64x64x200 and file_012.h5 -> 64x64x10.

Look at `train_solutions` which would allow access to the training validation and test sets.

`loss` is a typical loss function used to train a large neural network.
