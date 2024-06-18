class TrainingConfig:
    """
    Configuration class for training the model.

    Attributes:
    - image_size: the generated image resolution
    - train_batch_size: the batch size for training
    - eval_batch_size: the batch size for evaluation
    - num_epochs: the number of epochs to train
    - gradient_accumulation_steps: the number of steps to accumulate gradients before updating the model
    - learning_rate: the learning rate for the optimizer
    - lr_warmup_steps: the number of warmup steps for the learning rate scheduler
    - save_image_epochs: how often to save generated images
    - save_model_epochs: how often to save the model
    - mixed_precision: whether to use mixed precision training
    - output_dir: the directory to save the model and generated images
    - push_to_hub: whether to upload the saved model to the HF Hub
    - hub_model_id: the name of the repository to create on the HF Hub
    - hub_private_repo: whether to make the HF Hub repository private
    - overwrite_output_dir: whether to overwrite the old model when re-running the notebook
    - seed: the random seed for reproducibility
    """

    image_size = 64  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "prisoners-output"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = (
        "MGKK/prisonersi"  # the name of the repository to create on the HF Hub
    )
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


def getConfig() -> TrainingConfig:
    """
    Returns the training configuration.

    Returns:
    - TrainingConfig: the training configuration.
    """
    return TrainingConfig()
