import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from datasets.build import collate_function
from copy import deepcopy
from utils import is_dist_avail_and_initialized, get_world_size


class Trainer(nn.Module):
    def __init__(self,
                 pose_model,
                 criterion,
                 prompt_embeds_cfg,
                 prompt_momentum=0.9,
                 num_shots=1,
                 ft_steps=20,
                 freeze_encoder_in_ft=False):
        super().__init__()
        self.num_shots = num_shots
        self.ft_steps = ft_steps
        if num_shots == 1:
            self.ft_num_samples = dict(
                clothes=1,
                bird_body=1,
                mammal_body=1,
                animal_face=1,
                vehicle=4,
                furniture=4,
                insect_body=4,
                human_hand=4,
                human_face=4,
                human_body=4,
            )
        else:
            self.ft_num_samples = dict(
                clothes=1,
                bird_body=1,
                mammal_body=1,
                animal_face=1,
                vehicle=2,
                furniture=2,
                insect_body=2,
                human_hand=2,
                human_face=2,
                human_body=2
            )
        self.freeze_encoder_in_ft = freeze_encoder_in_ft

        self.pose_model = pose_model
        self.criterion = criterion

        self.prompt_embeds_dict = nn.ParameterDict()
        for item in prompt_embeds_cfg:
            self.prompt_embeds_dict[item] = nn.Parameter(torch.randn(*prompt_embeds_cfg[item]), requires_grad=False)
        self.prompt_momentum = prompt_momentum

    def embeds_update(self, prompt_embeds, features, joints, embeds_ids):
        sample_grid = joints[:, :, 0:-1] * 2 - 1
        sample_features = F.grid_sample(input=features,
                                        grid=sample_grid.unsqueeze(1),
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False).squeeze(2).permute(0, 2, 1)
        for n, (fea, weight) in enumerate(zip(sample_features, joints[:, :, -1])):
            ids_n = embeds_ids[n]
            fea = fea[:len(ids_n)]
            weight = weight[:len(ids_n)]
            current_embeds = prompt_embeds[ids_n]
            momentum = torch.ones([len(current_embeds)]).type_as(current_embeds) * self.prompt_momentum
            momentum[weight == 0] = 1.
            prompt_embeds[ids_n] = momentum[:, None] * current_embeds + (1 - momentum[:, None]) * fea

        return prompt_embeds

    def embeds_update_per_cat(self, prompt_embeds, features, joints, embeds_ids):
        sample_grid = joints[:, 0:-1] * 2 - 1
        sample_features = F.grid_sample(input=features.unsqueeze(0),
                                        grid=sample_grid.unsqueeze(0).unsqueeze(1),
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False).squeeze(2).squeeze(0).permute(1, 0)
        weight = joints[:, -1]
        current_embeds = prompt_embeds[embeds_ids]
        momentum = torch.ones([len(current_embeds)]).type_as(current_embeds) * self.prompt_momentum
        momentum[weight == 0] = 1.
        prompt_embeds[embeds_ids] = momentum[:, None] * current_embeds + (1 - momentum[:, None]) * sample_features

        return prompt_embeds

    def full_training(self, data, optimizer, **kwargs):
        images = data['image']
        joints = data['joints']
        heatmaps = data['target']
        heatmap_weights = data['target_weight']
        img_metas = data['img_metas']

        # stack inputs
        images = pad_sequence(images, batch_first=True)
        prompt_embeds = [
            self.prompt_embeds_dict[item['prompt_embedding_info']['type']][item['prompt_embedding_info']['ids']]
            for item in img_metas
        ]
        prompt_embeds = pad_sequence(prompt_embeds, batch_first=True)

        # stack targets
        heatmaps = pad_sequence(heatmaps, batch_first=True)
        heatmap_weights = pad_sequence(heatmap_weights, batch_first=True)

        # forward propagate
        predictions = self.pose_model(images, prompt_embeds)

        # compute loss
        losses = dict()
        for item in self.criterion:
            losses[item] = self.criterion[item](predictions, dict(heatmaps=heatmaps, heatmap_weights=heatmap_weights))
        total_loss = sum(losses.values())

        # backward update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # update prompt embeds
        for i, item in enumerate(img_metas):
            embed_ids = [item['prompt_embedding_info']['ids']]
            me_type = item['prompt_embedding_info']['type']
            update_embeds = self.embeds_update(self.prompt_embeds_dict[me_type].data,
                                               predictions['encode_fea'][i:i+1],
                                               torch.stack(joints[i:i+1], dim=0),
                                               embed_ids)
            self.prompt_embeds_dict[me_type].data = update_embeds

        if is_dist_avail_and_initialized():
            for item in self.prompt_embeds_dict:
                prompt_embeds = self.prompt_embeds_dict[item].data
                dist.all_reduce(prompt_embeds)
                self.prompt_embeds_dict[item].data = prompt_embeds / float(get_world_size())

        return losses

    def finetunning(self, data, optimizer, **kwargs):
        if self.freeze_encoder_in_ft:
            self.pose_model.freeze_stages(9)

        train_pipeline = deepcopy(kwargs['train_pipeline'])
        sampled_ids = list()
        for i in range(len(data)):
            task = data[i]['task']
            num_samples = self.ft_num_samples[task]
            sampled_ids += [i] * num_samples
        embeds_dict = dict()

        scaler = GradScaler() if self.num_shots > 1 else None

        # multi-step to finetune model via sup data
        for _ in range(self.ft_steps):
            sampled_data = [train_pipeline(deepcopy(data[id])) for id in sampled_ids]
            sampled_data = collate_function(sampled_data)
            prompt_embeds = [
                self.prompt_embeds_dict[item['prompt_embedding_info']['type']][item['prompt_embedding_info']['ids']]
                for item in sampled_data['img_metas']
            ]
            prompt_embeds = pad_sequence(prompt_embeds, batch_first=True)
            images = pad_sequence(sampled_data['image'], batch_first=True).type_as(prompt_embeds)
            heatmaps = pad_sequence(sampled_data['target'], batch_first=True).type_as(prompt_embeds)
            heatmap_weights = pad_sequence(sampled_data['target_weight'], batch_first=True).type_as(prompt_embeds)

            # forward
            if scaler is not None:
                with autocast():
                    predictions = self.pose_model(images, prompt_embeds)
                    losses = dict()
                    for item in self.criterion:
                        losses[item] = self.criterion[item](predictions, dict(heatmaps=heatmaps,
                                                                              heatmap_weights=heatmap_weights))
                    total_loss = scaler.scale(sum(losses.values()))
            else:
                predictions = self.pose_model(images, prompt_embeds)
                losses = dict()
                for item in self.criterion:
                    losses[item] = self.criterion[item](predictions, dict(heatmaps=heatmaps,
                                                                          heatmap_weights=heatmap_weights))
                total_loss = sum(losses.values())

            # backward
            optimizer.zero_grad()
            total_loss.backward()

            # step update
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # update embeds
            for features_n, joints_n, img_metas_n in zip(predictions['encode_fea'].detach(),
                                                         sampled_data['joints'],
                                                         sampled_data['img_metas']):

                prompt_type = img_metas_n['prompt_embedding_info']['type']
                embed_ids = img_metas_n['prompt_embedding_info']['ids']

                if prompt_type in embeds_dict:
                    prompt_embeds = embeds_dict[prompt_type]
                else:
                    prompt_embeds = self.prompt_embeds_dict[prompt_type]

                embeds_dict[prompt_type] = self.embeds_update_per_cat(prompt_embeds,
                                                                      features_n,
                                                                      joints_n.type_as(features_n),
                                                                      embed_ids)

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        return embeds_dict

    def forward(self, data, optimizer, phase, **kwargs):
        self.train()
        return getattr(self, phase)(data, optimizer, **kwargs)
