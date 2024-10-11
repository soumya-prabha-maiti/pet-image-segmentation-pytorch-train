import pytorch_lightning as pl

from pet_seg_train.config import EPOCHS, FAST_DEV_RUN
from pet_seg_train.data import train_dataloader
from pet_seg_train.model import UNet

trainer = pl.Trainer(max_epochs=EPOCHS, fast_dev_run=FAST_DEV_RUN)
model = UNet(3, 3)

trainer.fit(model, train_dataloader)
