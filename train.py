import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pet_seg_train.config import PetSegTrainConfig
from pet_seg_train.data import train_dataloader, val_dataloader
from pet_seg_train.model import UNet

def train():
    logger = CSVLogger("logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=PetSegTrainConfig.EPOCHS, fast_dev_run=PetSegTrainConfig.FAST_DEV_RUN, logger=logger, callbacks=[checkpoint_callback], gradient_clip_val=1.0
    )
    model = UNet(3, 3, depthwise_sep=PetSegTrainConfig.DEPTHWISE_SEP)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
