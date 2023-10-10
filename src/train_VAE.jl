using Flux
using Flux: throttle, params, Chain, Dense, pullback, MaxPool
using Flux: mse, logitbinarycrossentropy, binarycrossentropy, crossentropy, logitcrossentropy
using Parameters
using Parameters: @with_kw
using VideoIO
using FFMPEG
using Images
using Random
using CUDA
using Flux.Data: DataLoader
using BSON: @save, @load
using CairoMakie
using CairoMakie: Axis, Figure, lines!, scatter!, band!, image!
using Statistics: mean

include("training_parameters.jl")

struct Reshape
  shape
end

Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()


function load_data(; kws...)
  args = Args(; kws...)
  video_path = "../data/video/states/$(args.variation)_variation/states_$(args.data_set)_36x64.mp4"
  println(video_path)
  video = VideoIO.load(video_path)
  frames = size(video, 1)
  x_data = zeros(Float32, frames, args.input_dim[3], args.input_dim[1], args.input_dim[2])
  for i in 1:frames
    if args.input_dim[3] == 1
      x_data[i, 1, :, :] .= Float32.(channelview(Gray.(video[i])))
    elseif args.input_dim[3] == 3
      x_data[i, :, :, :] .= Float32.(channelview(RGB.(video[i])))
    end
  end

  X_train = permutedims(x_data, [3, 4, 2, 1])

  return X_train
end

function save_training_step(encoder, decoder, x, fig, ax_orig, ax_latent, ax_decoded, epoch, args)
  # save a image of the origonal data, latent space and the decoded image 
  mse_loss = Flux.mse(sigmoid.(decoder(encoder(x))[:, :, 1, 1]), x[:, :, 1, 1])
  empty!(ax_latent)
  image!(ax_orig, x[:, :, 1, 1])
  latent_space = encoder(x)
  lines!(ax_latent, latent_space[1, :])
  reconstructed_image = decoder(encoder(x[:, :, :, 1]))
  ax_orig.title = "min: $(minimum(x[:, :, 1, 1]))\n max: $(maximum(x[:, :, 1, 1]))\n avg: $(mean(x[:, :, 1, 1]))"
  ax_decoded.title = "MSE loss: $(mse_loss)\n min: $(minimum(sigmoid.(reconstructed_image[:, :, 1, 1])))\n max: $(maximum(sigmoid.(reconstructed_image[:, :, 1, 1])))\n avg: $(mean(sigmoid.(reconstructed_image[:, :, 1, 1])))"
  image!(ax_decoded, reconstructed_image[:, :, 1, 1])
  CairoMakie.save("training_step_VAE.pdf", fig)
end

function vae_train!(encoder_μ, encoder_logvar, decoder, model_loss, x_train, ps; kws...)

  args = Args(; kws...)
  args.seed > 0 && Random.seed!(args.seed)
  use_cuda = args.cuda && CUDA.functional()
  if use_cuda
    device = gpu
    println("Using GPU")
  else
    device = cpu
    println("Using CPU")
  end


  encoder_μ |> device
  encoder_logvar |> device
  decoder |> device

  len_data_loader = size(x_train, 4)
  data_loader = DataLoader(x_train, batchsize=len_data_loader, shuffle=true)

  opt = ADAM(args.η_vae)
  β = args.beta
  λ = args.λ_vae

  train_steps = 0
  train_save = zeros(args.epochs_vae)

  recon_loss = []

  # Plotting objects
  fig = Figure()
  ax_orig = CairoMakie.Axis(fig[1:3, 1:3])
  ax_latent = CairoMakie.Axis(fig[1:3, 4:6])
  ax_decoded = CairoMakie.Axis(fig[1:3, 7:9])
  for epoch = 1:args.epochs_vae

    n = 0
    l = 0
    for x in data_loader
      x |> device
      loss, back = pullback(ps) do
        model_loss(encoder_μ, encoder_logvar, decoder, x, β, λ)
      end
      grad = back(1.0f0)
      Flux.update!(opt, ps, grad)
      l += loss
    end

    avg_loss = l / len_data_loader

    train_save[epoch] = avg_loss
    if epoch % 20 == 0
      println("Epoch $epoch : Train loss: $(train_save[epoch]) ")
      recon = Flux.mse(sigmoid.(decoder(encoder_μ(x_train))), x_train) / size(x_train, 4)
      println("Epoch $epoch : Reconstruction loss: $(recon) ")
      append!(recon_loss, recon)
    end

    if epoch % 20 == 0
      let encoder_μ = cpu(encoder_μ), encoder_logvar = cpu(encoder_logvar), decoder = cpu(decoder)
        println("Saving model")
        @save "../models/$(args.variation)_variation/encoder_mu_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder_μ
        @save "../models/$(args.variation)_variation/encoder_logvar_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder_logvar
        @save "../models/$(args.variation)_variation/decoder_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" decoder
        # Save recon loss
        @save "../models/$(args.variation)_variation/recon_loss_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" recon_loss
        # Save vae_loss
        @save "../models/$(args.variation)_variation/vae_loss_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" train_save
        save_training_step(encoder_μ, decoder, x_train, fig, ax_orig, ax_latent, ax_decoded, epoch, args)
      end

    end
  end

  decoder |> cpu

  @save "../models/$(args.variation)_variation/encoder_mu_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder_μ
  @save "../models/$(args.variation)_variation/encoder_logvar_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder_logvar
  @save "../models/$(args.variation)_variation/decoder_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" decoder
  # Save recon loss
  @save "../models/$(args.variation)_variation/recon_loss_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" recon_loss
  # Save vae_loss
  @save "../models/$(args.variation)_variation/vae_loss_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" train_save
  #
end




function create_VAE(; kws...)
  args = Args(; kws...)

  pix_h = args.input_dim[1]
  pix_w = args.input_dim[2]
  channel = args.input_dim[3]
  Dz = args.latent_dim
  cnn_output_size = Int.(floor.([pix_w / 8, pix_h / 8, 32]))
  cnn_flat_size = 1024

  D_flat = 256

  conv_1 = Conv((args.filter_width, args.filter_width), channel => 32, tanh; stride=2, pad=1) #in [480, 640, 3, 71] out -> [240, 320, 32, 71] = [h/stride, w/stride, 32, b]
  conv_2 = Conv((args.filter_width, args.filter_width), 32 => 32, tanh; stride=2, pad=1) # in [240, 320, 32, 71], out -> [120, 160, 32, 71] = [(h/stride)/stride, (w/stride)/stride, 32, b]
  conv_3 = Conv((args.filter_width, args.filter_width), 32 => 32, tanh; stride=2, pad=1) # in [120, 240, 32, 71], out -> [60, 80, 32, 71] = [((h/stride)/stride)/stride, ((w/stride)/stride)/stride, 32, b]

  dense_in_1 = Dense(cnn_flat_size, D_flat, tanh)

  encoder_features = Chain(Flux.flatten, Dense(pix_h * pix_w * channel, cnn_flat_size, tanh), dense_in_1)

  encoder_μ = Chain(encoder_features, Dense(D_flat, Dz))
  encoder_logvar = Chain(encoder_features, Dense(D_flat, Dz))

  dense_out_1 = Dense(Dz, D_flat, tanh)
  dense_out_2 = Dense(D_flat, cnn_flat_size, tanh)

  deconv_1 = ConvTranspose((args.filter_width, args.filter_width), 32 => 32, tanh; stride=2, pad=(1, 1, 1, 1))
  deconv_2 = ConvTranspose((args.filter_width, args.filter_width), 32 => 32, tanh; stride=2, pad=(0, 0, 1, 1))
  deconv_out = ConvTranspose((args.filter_width, args.filter_width), 32 => channel; stride=2, pad=(1, 1, 1, 1))

  decoder = Chain(dense_out_1, dense_out_2, Reshape(cnn_output_size[2], cnn_output_size[1], 32, :), deconv_1, deconv_2, deconv_out)
  return encoder_μ, encoder_logvar, decoder
end



function vae_loss(encoder_μ, encoder_logvar, decoder, x, β, λ; kws...)
  batch_size = size(x)[end]
  @assert batch_size != 0
  μ = encoder_μ(x)
  logvar = encoder_logvar(x)

  # reparameterisation 
  z = μ + randn(Float32, size(logvar)) .* exp.(0.5f0 * logvar)

  x̂ = decoder(z)

  logp_x_z = -(logitbinarycrossentropy(x̂, x; agg=sum)) / batch_size

  kl_q_p = 0.5f0 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0f0)) / batch_size

  reg = λ * sum(x -> sum(x .^ 2), Flux.params(encoder_μ, encoder_logvar, decoder))

  elbo = logp_x_z - β .* kl_q_p

  return -elbo + reg
end


# --- Load model to further train --- #
function load_VAE(; kws...)
  args = Args(; kws...)
  @load "../models/$(args.variation)_variation/encoder_mu_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder_μ
  @load "../models/$(args.variation)_variation/encoder_logvar_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" encoder_logvar
  @load "../models/$(args.variation)_variation/decoder_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" decoder
  println("loaded VAE model")
  return encoder_μ, encoder_logvar, decoder
end

state_data = load_data()

# encoder_μ, encoder_logvar, decoder = load_VAE()
encoder_μ, encoder_logvar, decoder = create_VAE()

vae_ps = Flux.params(encoder_μ, encoder_logvar, decoder)
# function vae_train!(encoder_μ, encoder_logvar, model_loss, x_train, ps; kws...)
vae_train!(encoder_μ, encoder_logvar, decoder, vae_loss, state_data, vae_ps)



