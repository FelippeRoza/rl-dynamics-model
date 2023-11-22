

def get_base_config():
    
    return {
        # dynamics model configuration
        "dynamics_model": {
            "_target_": "costDynamicsModel.utils.GaussianMLPNew",
            "device": 'cpu',
            "num_layers": 3,
            "ensemble_size": 5,
            "hid_size": 200,
            "in_size": "???",
            "out_size": "???",
            "deterministic": True,
            "propagation_method": "fixed_model",
            # can also configure activation function for GaussianMLP
            "activation_fn_cfg": {
                "_target_": "torch.nn.LeakyReLU",
                "negative_slope": 0.01
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": False,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": 200,
            "num_steps": 10000
            ,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }