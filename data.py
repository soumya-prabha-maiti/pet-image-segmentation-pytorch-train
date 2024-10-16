import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms as T

from pet_seg_train.config import PetSegTrainConfig

# Define the transforms
transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
    ]
)

print(f"Downloading train_val samples")

# Download the dataset
train_val_ds = torchvision.datasets.OxfordIIITPet(
    root=PetSegTrainConfig.TRAIN_VAL_DATA_PATH,
    split="trainval",
    target_types="segmentation",
    transform=transform,
    target_transform=transform,
    download=True,
)

print(f"Downloaded train_val samples")

# Randomly sample some samples
train_val_ds = torch.utils.data.Subset(
    train_val_ds, torch.randperm(len(train_val_ds))[:PetSegTrainConfig.TRAIN_VAL_SAMPLES]
)

# Split the dataset into train val and test
train_ds, val_ds = torch.utils.data.random_split(
    train_val_ds,
    [int(0.8 * len(train_val_ds)), len(train_val_ds) - int(0.8 * len(train_val_ds))],
)

print(f"Downloading test samples")

test_ds, val_ds = torch.utils.data.random_split(
    val_ds,
    [int(0.5 * len(val_ds)), len(val_ds) - int(0.5 * len(val_ds))],
)
# test_ds = torchvision.datasets.OxfordIIITPet(
#     root=PetSegTrainConfig.TEST_DATA_PATH,#"./test_data",
#     split="test",
#     target_types="segmentation",
#     download=True,
# )

print(f"Downloaded test samples")

train_dataloader = DataLoader(
    train_ds,  # The training samples.
    sampler=RandomSampler(train_ds),  # Select batches randomly
    batch_size=PetSegTrainConfig.BATCH_SIZE,  # Trains with this batch size.
    num_workers=3,
    persistent_workers=True,
)

# For validation the order doesn't matter, so we'll just read them sequentially.
val_dataloader = DataLoader(
    val_ds,  # The validation samples.
    sampler=SequentialSampler(val_ds),  # Pull out batches sequentially.
    batch_size=PetSegTrainConfig.BATCH_SIZE,  # Evaluate with this batch size.
    num_workers=3,
    persistent_workers=True,
)

# For validation the order doesn't matter, so we'll just read them sequentially.
test_dataloader = DataLoader(
            test_ds, # The validation samples.
            sampler = SequentialSampler(test_ds), # Pull out batches sequentially.
            batch_size = PetSegTrainConfig.BATCH_SIZE, # Evaluate with this batch size.
            num_workers=3,
            persistent_workers=True,
        )
