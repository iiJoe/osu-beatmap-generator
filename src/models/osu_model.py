import constants
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding

class OsuModel(nn.Module):

    def __init__(self, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,dropout=0.1, max_seq_len=1024):
        super(OsuModel, self).__init__()
        input_dim = constants.input_dim
        output_dim = constants.cnn_output_dim

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        )

        self.positional_encoding = PositionalEncoding(output_dim, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.note_embedding = nn.Linear(constants.predictions_dim, output_dim, dtype=torch.float32)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Sequential(
            nn.Linear(output_dim, constants.predictions_dim, dtype=torch.float32),
            nn.Sigmoid()
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = src.permute(0, 2, 1)
        src = self.cnn_encoder(src)
        src = src.permute(0, 2, 1)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src, mask=src_mask)

        tgt = self.note_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_is_causal=True)
        output = self.out(output)

        return output
