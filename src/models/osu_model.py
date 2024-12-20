import constants
import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding

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

        self.encoder_positional_encoding = PositionalEncoding(output_dim, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.note_embedding = nn.Linear(constants.predictions_dim, output_dim)
        self.decoder_positional_encoding = PositionalEncoding(output_dim, dropout, max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(output_dim, constants.predictions_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.cnn_encoder(src.permute(0, 2, 1))
        src = src.permute(2, 0, 1)
        src = self.encoder_positional_encoding(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        tgt = self.note_embedding(tgt)
        tgt = tgt.permute(1, 0, 2)
        tgt = self.decoder_positional_encoding(tgt)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        output = self.out(output)
        output = output.permute(1, 0, 2)

        return output
