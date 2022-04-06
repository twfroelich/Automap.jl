using CSV, DataFrames
using ImageMagick, Images, FFTW, HDF5, Metalhead
using DataSets
import FileIO
using JpegTurbo
using .Threads

function minibatch(data_tree, key; nsamples = 16, dataset="singlecoil_train", image_size = (368÷2,640÷2), kwargs...)
  s = @view key[rand(1:size(key,1), nsamples), :]
  @views begin
    arr_kspace = [] # Kspace
    arr_image = [] # Image

    ps = makepaths.(s.ImageId, dataset)
    @sync for (i,p) in enumerate(ps)
      Threads.@spawn fproc(data_tree,arr_kspace,arr_image,p,image_size)
    end
  end
  return reduce(hcat,arr_kspace), reduce(hcat,arr_image)
end
function fproc(data_tree,dest_kspace,dest_image,path,size_out_mat)
  x = open(IO, data_tree[path]) do io
    h5read(chop(string(io),head=15,tail=2),"kspace")
  end

  recon_tmp = Array{Float32}(undef,size_out_mat[1],size_out_mat[2],size(x,3))
  kspace_tmp = Array{ComplexF32}(undef,size_out_mat[1],size_out_mat[2],size(x,3))
  for index = 1:size(x,3)
    recon = imresize(abs.(FFTW.ifftshift(FFTW.fft(x[:,:,index]))),(size_out_mat[1],size_out_mat[2]))

    recon_tmp[:,:,index] = recon
    kspace_tmp[:,:,index] = FFTW.fftshift(FFTW.ifft(FFTW.fftshift(recon)))
  end
  k_tmp = reshape(kspace_tmp,(size(kspace_tmp,1)*size(kspace_tmp,2),1,size(kspace_tmp,3)))

  push!(dest_kspace,Float32.(reshape((cat(real.(k_tmp),imag.(k_tmp),dims=2)),(size(k_tmp,1)*2,size(k_tmp,3)))))
  push!(dest_image,Float32.(reshape(recon_tmp,(size(recon_tmp,1)*size(recon_tmp,2),size(recon_tmp,3)))))
end
function makepaths(imgs, dataset, base = ["Data"])
  if dataset == "singlecoil_train"
    return DataSets.RelPath([base..., dataset, imgs * ".h5"])
  elseif dataset == "singlecoil_val"
    return DataSets.RelPath([base..., dataset, imgs * ".h5"])
  elseif dataset == "singlecoil_test_v2"
    return DataSets.RelPath([base..., dataset, imgs * ".h5"])
  end
end
function train_solutions(data_tree, train_sol_file = path"train_data_key.csv")
  df = open(IO, data_tree[train_sol_file]) do io
    CSV.File(io) |> DataFrame
  end
end
function val_solutions(data_tree, val_sol_file = path"val_data_key.csv")
  df = open(IO, data_tree[val_sol_file]) do io
    CSV.File(io) |> DataFrame
  end
end
