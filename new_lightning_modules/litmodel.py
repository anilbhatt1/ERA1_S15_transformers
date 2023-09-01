import torch
import torch.nn as nn
import torchmetrics
import lightning as L
import numpy 

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, tokenizer_src, tokenizer_tgt, max_len, num_examples):
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
        self.num_examples = num_examples   
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
        if batch_idx < self.num_examples:
            encoder_input = batch['encoder_input']
            encoder_mask = batch['encoder_mask']

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, self.max_len)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            self.source_texts.append(source_text)
            self.expected.append(target_text)
            self.predicted.append(model_out_text)

        if batch_idx >= self.num_examples:
            metric = torchmetrics.CharErrorrate()
            cer = metric(self.predicted, self.expected)
            self.log("val_cer", cer)

            metric = torchmetrics.WordErrorRate()
            wer = metric(self.predicted, self.expected)
            self.log("val_wer", wer)   

            metric = torchmetrics.BLEUScore()
            bleu = metric(self.predicted, self.expected) 
            self.log("val_bleu", bleu)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-9)
        return optimizer