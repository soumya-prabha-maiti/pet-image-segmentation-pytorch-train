import pytorch_lightning as pl

from pet_seg_train.config import PetSegTrainConfig
from pet_seg_train.data import train_dataloader
from pet_seg_train.model import UNet

trainer = pl.Trainer(
    max_epochs=PetSegTrainConfig.EPOCHS, fast_dev_run=PetSegTrainConfig.FAST_DEV_RUN
)
model = UNet(3, 3)

trainer.fit(model, train_dataloader)
