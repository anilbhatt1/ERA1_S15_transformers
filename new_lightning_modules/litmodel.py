import torch
import torch.nn as nn
import torchmetrics
import lightning as L
import numpy 
from dataset import causal_mask

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, tokenizer_src, tokenizer_tgt, max_len):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
        self.max_len = max_len
        self.source_texts = []
        self.expected = []
        self.predicted = [] 
        self.cer_metric = torchmetrics.text.CharErrorRate()
        self.wer_metric = torchmetrics.text.WordErrorRate()
        self.bleu_metric = torchmetrics.text.BLEUScore()
        self.save_hyperparameters(ignore=['model'])   

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.model.project(decoder_output)
        label = batch['label']
        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval() 
        encoder_input = batch['encoder_input']
        encoder_mask = batch['encoder_mask']

        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = self.greedy_decode(encoder_input, encoder_mask)
        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)

    def greedy_decode(self, source, source_mask):
        sos_idx = self.tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = self.tokenizer_tgt.token_to_id("[EOS]")

        # Precompute the encoder output and reuse it for every step
        encoder_output = self.model.encode(source, source_mask)
        # Initialize the decoder input with the start of sentence token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)

        while True:
            if decoder_input.size(1) == self.max_len:
                break
            
            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask)

            # calculate output
            out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = self.model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item())], dim=1)
            
            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)

    def test_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        cer = self.cer_metric(self.predicted, self.expected)
        self.log("val_cer", cer, prog_bar=True, on_epoch=True)

        wer = self.wer_metric(self.predicted, self.expected)
        self.log("val_wer", wer, prog_bar=True, on_epoch=True)   

        bleu = self.bleu_metric(self.predicted, self.expected) 
        self.log("val_bleu", bleu, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-9)
        return optimizer