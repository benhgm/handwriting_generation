import torch
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from utils import draw, generate_text_from_onehots, attention_plot, draw_dataset2
from modules import bivariate_gaussian_nll
from model import HandwritingSynthesisNetwork

from PIL import Image

class HandwritingAuthorAdaptation(LightningModule):
    def __init__(self, lr, optim, dictionary_size):
        super().__init__()

        self.model = HandwritingSynthesisNetwork(dictionary_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.lr = lr
        self.optim = optim
    
    def forward(self, x, text_mask, onehots, hidden_states):
        return self.model(x, text_mask, onehots, hidden_states)
    
    def training_step(self, batch, batch_idx):
        stroke, stroke_mask, onehots, text_mask, text_len = batch
        input_strokes = stroke[:, :-1, :]
        
        target_strokes = stroke[:, 1:, :]
        target_stroke_masks = stroke_mask[:, :-1]

        pi, mu, sigma, rho, eos, w_t, kappa, phi, h0, c0, h1, c1 = self(input_strokes, onehots, text_mask, hidden_states=None)

        nll_loss = bivariate_gaussian_nll(pi, mu, sigma, rho, eos, target_strokes, target_stroke_masks)
        nll_loss = nll_loss / torch.sum(target_stroke_masks)

        self.log("train_loss", nll_loss, on_step=False, on_epoch=True)

        return {"loss": nll_loss}
    
    def validation_step(self, batch, batch_idx):
        stroke, stroke_mask, onehots, text_mask, text_len = batch
        input_strokes = stroke[:, :-1, :]
        
        target_strokes = stroke[:, 1:, :]
        target_stroke_masks = stroke_mask[:, 1:]

        pi, mu, sigma, rho, eos, w_t, kappa, phi, h0, c0, h1, c1 = self(input_strokes, onehots, text_mask, hidden_states=None)

        nll_loss = bivariate_gaussian_nll(pi, mu, sigma, rho, eos, target_strokes, target_stroke_masks)
        nll_loss = nll_loss / torch.sum(target_stroke_masks)

        with torch.no_grad():
            out, phis = self.model.sample(onehots=onehots, text_mask=text_mask, maxlen=2000)

            out = out.cpu().numpy()
            phis = phis.cpu().numpy()

        # Create figures for logging and visualisation of training progress
        sample_text_len = int(text_len[0].item())
        draw(out[0], save_file='./generated.jpg')
        attention_plot(phis[0][:, :sample_text_len])

        del out
        del phis

        # Open images and generate caption
        log_img = Image.open('./generated.jpg')
        log_atn = Image.open('./attention_plot.jpg')
        text_caption = generate_text_from_onehots(onehots, sample_text_len)

        # Log images and losses
        self.logger.log_image(key="validation samples", images=[log_img, log_atn], caption=[text_caption, "attention"])
        self.log("val_loss", nll_loss, on_step=False, on_epoch=True)

        return {"loss": nll_loss}
        
    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = Adam(self.parameters(), lr=self.lr)
        if self.optim == "sgd":
            optimizer = SGD(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer, mode="min", threshold=0.05, verbose=True),
            "monitor": "val_loss",
            "frequency": 1
        }

if __name__ == "__main__":
    model = HandwritingAuthorAdaptation(0.001, "adam")
    print(model)