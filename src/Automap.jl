module Automap

using Flux, CUDA, FFTW, HDF5, Images
using Metalhead
using BSON, Zygote
using Distributed
using Dates, DataSets
using Functors, Optimisers
using Requires, Logging

export minibatch, train_solutions, syncgrads
export prepare_training

include("utils.jl")
include("overloads.jl")
include("automap.jl")
include("ddp_tasks.jl")

@require Wandb="ad70616a-06c9-5745-b1f1-6a5f42545108" begin
  include("loggers/wandb.jl")
end

end # module
