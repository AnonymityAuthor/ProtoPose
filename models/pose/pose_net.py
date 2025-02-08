import torch.nn as nn


class ProtoPose(nn.Module):
    def __init__(self,
                 encoder,
                 decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_attn(self):
        self.encoder.freeze_attn()

    def freeze_ffn(self):
        self.encoder.freeze_ffn()

    def freeze_stages(self, frozen_stages):
        self.encoder.freeze_stages(frozen_stages)

    def get_parameters(self):
        fast_weights = dict()

        encoder_params = dict(self.encoder.get_parameters())
        for item in encoder_params:
            fast_weights['encoder.{}'.format(item)] = encoder_params[item]

        decoder_params = dict(self.decoder.named_parameters())
        for item in decoder_params:
            fast_weights['decoder.{}'.format(item)] = decoder_params[item]

        return fast_weights

    def forward(self, image, prompt_embeds=None, fast_weights=None):
        if prompt_embeds is None:
            xp = self.encoder(image)
            return dict(
                encode_fea=xp,
            )
        else:
            encoder_weights = dict()
            decoder_weights = dict()
            if fast_weights:
                for item in fast_weights:
                    if 'encoder.' in item:
                        new_item = item.replace('encoder.', '')
                        encoder_weights[new_item] = fast_weights[item]
                    elif 'decoder.' in item:
                        new_item = item.replace('decoder.', '')
                        decoder_weights[new_item] = fast_weights[item]

            xp, inc_embeds = self.encoder(image, prompt_embeds, encoder_weights)
            heatmaps, x_up = self.decoder(xp, inc_embeds, decoder_weights)

            return dict(
                encode_fea=x_up,
                heatmaps=heatmaps
            )
