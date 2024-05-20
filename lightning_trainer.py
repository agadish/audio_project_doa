import lightning as L
import torch
import torch.nn.functional as F
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
        target = target.to(self.device, dtype=torch.long)

        # TODO: Train the RNN model on one batch of data.
        outputs = self.model(samples)
        # TODO
        # output_directions = torch.dot(outputs, ref_stft * ref_stft.T)
        # output_angle = torch.argmax(output_directions, axis=1)
        loss = self.loss_fn(outputs, target // ANGLE_RES)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitor validation loss for adjusting LR
            }
        }
    
    def validation_step(self, batch, batch_idx):
        samples, target = batch
        samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
        target = target.to(self.device, dtype=torch.long)

        outputs = self.model(samples)
        loss = self.loss_fn(outputs, target // ANGLE_RES)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
     
        return loss

    def test_step(self, batch, batch_idx):
        samples, ref_stft, target, mixed_signals, perceived_signals, radii, reverbs = batch
        n_samples = samples.size(0)

        samples = samples.to(self.device, dtype=torch.float)  # (B,S,V)
        target = target.to(self.device, dtype=torch.long)
        probs = self.model(samples)
        probs = F.softmax(probs, dim=1) # dim=1 refers to the 13 possible DOAs
        radii = radii.to(self.device)
        reverbs = reverbs.to(self.device)

        batch_dict = {
            'ref_stft': ref_stft,
            'mixed_signals': mixed_signals,
            'perceived_signals': perceived_signals,
            'probs': probs
        }
        
        batch_mix = mixed_batch_metrics(batch_dict)
        batch_sep = separated_batch_metrics(batch_dict)

        avg_mix_groups = {}
        avg_sep_groups = {}

        for i in range(n_samples):
            key = f"rad{radii[i][0].item()}_rev{reverbs[i][0].item()}" # Assuming [0] same as [1]
            avg_mix_groups.setdefault(f"mix_{key}", []).append(batch_mix[i])
            avg_sep_groups.setdefault(f"sep_{key}", []).append(batch_sep[i])
        
        avg_mix = {group: torch.mean(torch.stack(values), axis=0) for group, values in avg_mix_groups.items()}
        avg_sep = {group: torch.mean(torch.stack(values), axis=0) for group, values in avg_sep_groups.items()}

        for group_name, value in avg_mix.items():
            for i in range(value.size(0)):
                for j in range(value.size(1)):
                    self.log(f'{group_name}_{i}_{j}', value[i, j].item(), prog_bar=True, on_step=True, on_epoch=True)

        for group_name, value in avg_sep.items():
            for i in range(value.size(0)):
                for j in range(value.size(1)):
                    self.log(f'{group_name}_{i}_{j}', value[i, j].item(), prog_bar=True, on_step=True, on_epoch=True)

        result = avg_mix.copy()
        result.update(avg_sep)
        return result
    