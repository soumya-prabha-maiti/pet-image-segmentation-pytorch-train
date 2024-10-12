from dataclasses import dataclass

@dataclass
class PetSegTrainConfig:
    EPOCHS = 5
    BATCH_SIZE = 8
    FAST_DEV_RUN = False
    TRAIN_VAL_SAMPLES= 100
    LEARNING_RATE = 1e-3
    TRAIN_VAL_DATA_PATH = "./data/train_val"
    TEST_DATA_PATH = "./data/test"