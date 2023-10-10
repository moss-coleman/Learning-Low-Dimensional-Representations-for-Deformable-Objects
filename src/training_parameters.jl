using Parameters: @with_kw

@with_kw mutable struct Args
  η_vae = 1e-4                # VAE learning rate
  η_mlp = 1e-5                # MLP learning rate
  λ_vae = 0.01f0              # VAE regularization paramater
  λ_mlp = 0.01f0              # MLP regularization paramater
  epochs_vae = 20000          # number of epochs
  epochs_mlp = 30000          # number of epochs
  seed = 42                   # random seed
  cuda = true                 # use GPU
  input_dim = [36, 64, 1]     # image size
  mlp_input_dim = 20          # MLP input dimension 
  latent_dim = 1              # latent dimension
  hidden_layers = 20          # latent dimension
  beta = 1.0                  # β value in loss function
  variation = "length"        # what to vary, "shape" or "length"
  data_set = "150"            # what set of states to train the VAE to represent
  filter_width = 4            # CNN kernal dimension
  resize_ratio = 10           # downsampling ratio of images 
  fps = 240.0                 # frames per second of video
  encoder_variation = "vae"
end

