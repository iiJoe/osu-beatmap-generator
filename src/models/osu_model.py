import constants
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class OsuModel(nn.Module):

    def __init__(self, nhead=8, n_layers=8, d_model=1024, dim_feedforward=2048, dropout=0.1):
        super(OsuModel, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, 512)
        self.src_fc = nn.Linear(constants.input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.positional_encoding_tgt = PositionalEncoding(d_model, 120)
        self.note_embedding = nn.Linear(constants.predictions_dim, d_model, dtype=torch.float32)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, constants.predictions_dim, dtype=torch.float32),
            nn.Sigmoid()
        )

    def forward(self, src, tgt, spec_pad_mask=None, hit_pad_mask=None):
        spec_pad_mask = spec_pad_mask.to(src.device)
        hit_pad_mask = hit_pad_mask.to(tgt.device)

        src = torch.permute(src, (0, 2, 1))
        src = self.src_fc(src)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=spec_pad_mask)

        from utils import causal_mask

        tgt_mask = causal_mask(tgt.shape[1]).to(tgt.device)
        tgt = self.note_embedding(tgt)
        tgt = self.positional_encoding_tgt(tgt)
        tgt = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=spec_pad_mask, tgt_key_padding_mask=hit_pad_mask)

        output = self.out(tgt)

        return output
