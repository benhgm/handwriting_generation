# Handwriting Generation with Recurrent Neural Networks
This repository contains a PyTorch implementation of Alex Graves' paper: [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850).

The code in this repository is heavily referenced from the following repositories:
- [Handwriting Generation](https://github.com/wezteoh/handwriting_generation.git)
- [Reproducing handwriting synthesis from the seminal paper - Generating Sequences Using Recurrent Neural Networks by Alex Graves](https://github.com/ritheshkumar95/pytorch_handwriting_synthesis.git)


# Training Config
A sample training config is shown below. The training run is logged using [Weights & Biases](https://wandb.ai).

### Config Variables
#### Weights & Biases Config
`project_name`: Name of the project <br>
`group_name`: The experiment group name <br>
`run_name`: Name of the current experiment run <br>

#### Hyperparameters / Parameters
`batch_size`: Batch size<br>
`epochs`: Number of epochs<br>
`lr`: Initial learning rate<br>
`optim`: Optimizer to use<br>
`dictionary_size`: Number of alphabetical/symbolic characters to predict for<br>
`checkpoint_dir`: Checkpoint save directory<br>

```
{
    "group_name": "handwriting_synthesis",
    "run_name": "test_30_mixture_component",
    "batch_size": 50,
    "epochs": 60,
    "lr": 0.0005,
    "optim": "adam",
    "dictionary_size": 60,

    "checkpoint_dir": "experiments/",

    "project_name": "handwriting_generation"
}
```
# Training
To train a handwriting generation model, execute the following command:
```
python train.py
```

# Generation
To generate a sample handwritten text, edit the input text in `demo.py` and execute:
```
python demo.py
```