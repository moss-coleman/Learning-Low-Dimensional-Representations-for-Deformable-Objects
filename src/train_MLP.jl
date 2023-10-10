using Revise
using Flux: throttle, params, Chain, Dense
using Flux, Statistics
using Flux.Data: DataLoader
using Random
using CUDA
using Parameters: @with_kw
using BSON: @save, @load

include("training_parameters.jl")

function load_encoded_data(; kws...)

  args = Args(; kws...)
  if args.encoder_variation == "vae"
    @load "../data/encoded_time_series/$(args.variation)_variation/x_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_train
    @load "../data/encoded_time_series/$(args.variation)_variation/x_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_test
    @load "../data/encoded_time_series/$(args.variation)_variation/y_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_train
    @load "../data/encoded_time_series/$(args.variation)_variation/y_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_test
  elseif args.encoder_variation == "ae"
    @load "../data/encoded_time_series/$(args.variation)_ae_variation/x_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_train
    @load "../data/encoded_time_series/$(args.variation)_ae_variation/x_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" x_test
    @load "../data/encoded_time_series/$(args.variation)_ae_variation/y_train_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_train
    @load "../data/encoded_time_series/$(args.variation)_ae_variation/y_test_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" y_test
  end
  return x_train, x_test, y_train, y_test
end


x_train, x_test, y_train, y_test = load_encoded_data()
x_train = reshape(x_train, size(x_train)[1] * size(x_train)[2], size(x_train)[3])
x_test = reshape(x_test, size(x_test)[1] * size(x_test)[2], size(x_test)[3])

function mlp_train!(model, model_loss, x_data, y_data, ps; kws...)
  args = Args(; kws...)

  args.seed > 0 && Random.seed!(args.seed)
  use_cuda = args.cuda && CUDA.functional()

  if use_cuda
    device = gpu
    @info "Training on GPU"
  else
    device = cpu
    @info "Training on CPU"
  end

  opt = Flux.Adam(args.η_mlp)

  @info "MLP: $(num_params(model)) trainable params"

  train_save = zeros(args.epochs_mlp)

  data_loader = DataLoader((x_data, y_data), batchsize=128, shuffle=true)
  @info "test for DataLoader"
  for epoch = 1:args.epochs_mlp
    n = 0
    l = 0

    for (x, y) in data_loader
      x |> device
      y |> device
      loss, back = Flux.pullback(ps) do
        model_loss(x, y)
      end
      grad = back(1.0f0)
      Flux.update!(opt, ps, grad)
      l += loss
      n += size(x)[end]
    end
    train_save[epoch] = l / n

    if epoch % 100 == 0
      @info "Epoch $epoch : Train loss = $(train_save[epoch]) "
    end

  end

  model |> cpu
  if args.encoder_variation == "vae"
    @save "../models/$(args.variation)_variation/MLP/mlp_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" model
  elseif args.encoder_variation == "ae"
    @save "../models/$(args.variation)_ae_variation/MLP/mlp_$(args.data_set)_beta_$(floor(Int, args.beta))-$(Int(10*args.beta)%10)_dz_$(args.latent_dim).bson" model
  end

end

function create_MLP(; kws...)
  args = Args(; kws...)


  mlp = Chain(Dense(args.mlp_input_dim * args.latent_dim, args.hidden_layers, tanh),
    Dense(args.hidden_layers, args.hidden_layers, tanh),
    Dense(args.hidden_layers, args.hidden_layers, tanh),
    Dense(args.hidden_layers, args.hidden_layers, tanh),
    Dense(args.hidden_layers, args.latent_dim))

  return mlp
end


num_params(model) = sum(length, Flux.params(model))

mse_loss(x, y) = Flux.mse(mlp(x), y)

mlp = create_MLP()

ps = Flux.params(mlp)

mlp_train!(mlp, mse_loss, x_train, y_train, ps)
