using Automap
using Automap.BSON
using Automap.Metalhead
using Automap.Flux
using Automap.DataSets

using Distributed
@everywhere using Flux, BSON, CUDA, Metalhead, Zygote, Distributed, DataSets, Automap, HDF5, FFTW, Images
