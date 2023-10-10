# using Revise
using VideoIO
using Flux
using Flux: throttle, params, Chain, Dense, unsqueeze
using Flux.Data: DataLoader
using BSON: @save, @load
using CSV
using MAT
using Parameters
using Parameters: @with_kw
using Images
using Statistics
using CairoMakie

include("training_parameters.jl")

# --- Load VAE Models --- # 
struct Reshape
  shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()

function load_encoder(; kws...)
  args = Args(; kws...)

  if args.encoder_variation == "vae"
    @load "../models/$(args.variation)_variation/encoder_mu_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder_μ
    encoder = encoder_μ
  elseif args.encoder_variation == "ae"
    @load "../models/$(args.variation)_ae_variation/encoder_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder
  end
  return encoder
end

function video2time_series(video, encoder; kws...)

  args = Args(; kws...)
  resize_ratio = args.resize_ratio
  frames = size(video, 1)
  pix_h = args.input_dim[1]
  pix_w = args.input_dim[2]
  channel = args.input_dim[3]
  latent_dim = args.latent_dim
  video_resized = zeros(Float32, channel, pix_h, pix_w, frames)
  time_series_whole = zeros(Float32, frames, latent_dim)
  for i in 1:frames
    if args.input_dim[3] == 1
      img = Float32.(channelview(Gray.(imresize(video[i], ratio=1 / resize_ratio))))
      video_resized[1, :, :, i] .= img
    elseif args.input_dim[3] == 3
      img = Float32.(channelview(RGB.(imresize(video[i], ratio=1 / resize_ratio))))
      video_resized[:, :, :, i] .= img
    end
  end
  println("size(video_resized)", size(video_resized))
  # without the permutedims, the array was filling up by interleaving instead of stacking 
  println("encoder(video_resized): ", size(encoder(video_resized)))
  time_series_whole[:, :] = permutedims(encoder(video_resized), [2, 1])
  println("size(time_series_whole)", size(time_series_whole))

  # create the training data 
  sample_interval = 1
  dt = (1.0 / args.fps) * sample_interval
  output_window = 1
  input_dim = args.mlp_input_dim

  # time_series = time_series_whole[begin:sample_interval:end, :]
  time_series = time_series_whole#[begin:sample_interval:end, :]
  println("test 1")
  println("time_series_whole: ", size(time_series_whole))
  x_data = time_series[1:(end-output_window), :]
  println("x_data: ", size(x_data))
  y_data = time_series[(1+input_dim):end, :]
  y_video = video_resized[:, :, :, (1+input_dim):end]
  println("size y_data: ", size(y_data))
  println("size y_video: ", size(y_video))
  x_data_win = zeros((size(x_data, 1) - input_dim + 1), input_dim, latent_dim)
  println("x_data_win: ", size(x_data_win))
  println("test 2")

  for j in 1:(size(x_data, 1)-input_dim+1)
    x_data_win[j, :, :] = x_data[j:(j+input_dim-1), :]
  end
  println("test 3")

 return time_series[1:end-1, :], x_data_win[1:end-1, :, :], y_data[1:end-1, :], y_video[:, :, :, 1:end-1]
end

function save_data(x_train, x_test, y_train, y_test, y_test_video; kws...)
  args = Args(; kws...)
  if args.encoder_variation == "vae"
    @save "../data/encoded_time_series/$(args.variation)_variation/x_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_train
    @save "../data/encoded_time_series/$(args.variation)_variation/x_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_test
    @save "../data/encoded_time_series/$(args.variation)_variation/y_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_train
    @save "../data/encoded_time_series/$(args.variation)_variation/y_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_test

    @save "../data/encoded_time_series/$(args.variation)_variation/y_test_video_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10).bson" y_test_video

    file = matopen("../data/encoded_time_series/$(args.variation)_variation/x_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "x_train", x_train)
    MAT.close(file)

    file = matopen("../data/encoded_time_series/$(args.variation)_variation/x_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "x_test", x_test)
    MAT.close(file)

    file = matopen("../data/encoded_time_series/$(args.variation)_variation/y_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "y_train", y_train)
    MAT.close(file)

    file = matopen("../data/encoded_time_series/$(args.variation)_variation/y_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "y_test", y_test)
    MAT.close(file)
  elseif args.encoder_variation == "ae"

    @save "../data/encoded_time_series/$(args.variation)_ae_variation/x_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_train
    @save "../data/encoded_time_series/$(args.variation)_ae_variation/x_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_test
    @save "../data/encoded_time_series/$(args.variation)_ae_variation/y_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_train
    @save "../data/encoded_time_series/$(args.variation)_ae_variation/y_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_test

    @save "../data/encoded_time_series/$(args.variation)_ae_variation/y_test_video_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10).bson" y_test_video

    file = matopen("../data/encoded_time_series/$(args.variation)_ae_variation/x_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "x_train", x_train)
    MAT.close(file)

    file = matopen("../data/encoded_time_series/$(args.variation)_ae_variation/x_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "x_test", x_test)
    MAT.close(file)

    file = matopen("../data/encoded_time_series/$(args.variation)_ae_variation/y_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "y_train", y_train)
    MAT.close(file)

    file = matopen("../data/encoded_time_series/$(args.variation)_ae_variation/y_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).mat", "w")
    MAT.write(file, "y_test", y_test)
    MAT.close(file)
  end
end

function load_video(n; kws...)
  args = Args(; kws...)
  video = VideoIO.load("../data/video/time_series/$(args.variation)_variation/$(args.data_set)_$(n).mp4")
  return video
end

# --- Load Encoder --- #
println("Loading Encoder")
encoder = load_encoder()


train_video_no = 4
test_video = 2

function plot_encoding(z, n)
  f = Figure()
  ax = GLMakie.Axis(f[1, 1])
  lines!(ax, z, color=:blue, label="z")
  save("z_$(n).png", f)
end


for i in 1:train_video_no
  println("Loading Video $(i)")
  video = load_video(i)
  println("Encoding Video $(i)")
  time_series, x_data_win, y_data, y_video = video2time_series(video, encoder)
  println("test 1")
  if i == 1
    global x_train = x_data_win
    global y_train = y_data
  elseif i == test_video
    global x_test = x_data_win
    global y_test = y_data
    global y_test_video = y_video
  else
    x_train = vcat(x_train, x_data_win)
    y_train = vcat(y_train, y_data)
  end
end

x_train = permutedims(x_train, [2, 3, 1])
x_test = permutedims(x_test, [2, 3, 1])
y_train = permutedims(y_train, [2, 1])
y_test = permutedims(y_test, [2, 1])

save_data(x_train, x_test, y_train, y_test, y_test_video)
