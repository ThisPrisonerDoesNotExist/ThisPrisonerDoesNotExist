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
    - download_data_dir: the directory to download the training data
    - training_data_dir: the directory containing the training data
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
    output_dir = "prisoners-output"
    download_data_dir = "data/training data"
    training_data_dir = "C:/COV/ThisPrisonerDoesNotExist/thisprisonerdoesnotexist/data/training data/front"  # the directory containing the training data
    test_data_dir = "../../data/example train data"
    dataset_url = "https://huggingface.co/datasets/MGKK/Prisonersi"
    seed = 0
