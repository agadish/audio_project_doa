import lightning as L
import torch
from config import ANGLE_RES
from metrics import mixed_batch_metrics, separated_batch_metrics


class UnetDACLighting(L.LightningModule):
    def __init__(self, model, loss_fn, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        samples, target = batch
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
        samples, target = batch
        samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
        target = target.to(self.device)

        outputs = self.model(samples)
        loss = self.loss_fn(outputs, target // ANGLE_RES)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
     
        return loss

    def test_step(self, batch, batch_idx):
        samples, ref_stft, target, mixed_signals, perceived_signals = batch

        samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
        target = target.to(self.device)
        probs = self.model(samples)

        batch_dict = {
            'ref_stft': ref_stft,
            'mixed_signals': mixed_signals,
            'perceived_signals': perceived_signals,
            'probs': probs
        }
        
        batch_mix = mixed_batch_metrics(batch_dict)
        batch_sep = separated_batch_metrics(batch_dict)
        avg_mix = batch_mix.mean(dim=0)
        avg_sep = batch_sep.mean(dim=0)
        for i in range(avg_mix.size(0)):
            for j in range(avg_mix.size(1)):
                self.log(f'avg_mix_{i}_{j}', avg_mix[i, j].item(), prog_bar=True, on_step=True, on_epoch=True)

        for i in range(avg_sep.size(0)):
            for j in range(avg_sep.size(1)):
                self.log(f'avg_sep_{i}_{j}', avg_sep[i, j].item(), prog_bar=True, on_step=True, on_epoch=True)
        
        return {'avg_mix': avg_mix, 'avg_sep': avg_sep}