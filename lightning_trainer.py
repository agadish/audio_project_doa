import lightning as L
import torch
from config import ANGLE_RES


class UnetDACLighting(L.LightningModule):
    def __init__(self, model, loss_fn, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        samples, _, target = batch
        samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
        target = target.to(self.device)

        # TODO: Train the RNN model on one batch of data.
        outputs = self.model(samples)
        # TODO
        # output_directions = torch.dot(outputs, ref_stft * ref_stft.T)
        # output_angle = torch.argmax(output_directions, axis=1)
        loss = self.loss_fn(outputs, target // ANGLE_RES)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitor validation loss for adjusting LR
                'mode': 'min',  # Minimize validation loss
                'factor': 0.1,  # Multiply LR by 0.1 when triggered
                'patience': 10,  # Wait for 10 epochs before reducing LR
                'verbose': True  # Print updates
            }
        }
    
    def validation_step(self, batch, batch_idx):
        samples, _, target = batch
        samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
        target = target.to(self.device)

        outputs = self.model(samples)
        loss = self.loss_fn(outputs, target // ANGLE_RES)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
     
        return loss
