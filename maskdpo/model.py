from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.parallel.sequence import (gather_forward_split_backward,
                                      get_sequence_parallel_group,
                                      get_sequence_parallel_world_size,
                                      split_for_sequence_parallel)
from xtuner.model.dpo import DPO

class MaskDPO(DPO):
    
    def compute_loss(self, data, data_samples=None):
        # modified from https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py  # noqa
        # shift labels first and add a dummy label at the end, to support sequence parallel  # noqa
        data['labels'] = torch.cat(
            (data['labels'][:, 1:], torch.zeros_like(data['labels'][:, :1])),
            dim=1)
        tmp_label = data['labels'].clone()
        tmp_label[tmp_label == 0] = -100
        all_loss_mask = data[
            'labels'] != -100  # loss mask of all tokens in all sp ranks  # noqa

        if get_sequence_parallel_world_size() > 1:
            data = self._split_for_sequence_parallel(data)

        all_logits = self.llm(**data).logits
        with torch.no_grad():
            if self.ref_llm is None:
                with self.llm.disable_adapter():
                    all_ref_logits = self.llm(**data).logits
            else:
                all_ref_logits = self.ref_llm(**data).logits

        labels = data['labels']
        labels[labels == -100] = 0
        loss_mask = labels != 0  # loss mask in a single sp rank
        policy_logps = self._gather_masked_logits(all_logits, labels,
                                                  loss_mask)
        ref_logps = self._gather_masked_logits(all_ref_logits, labels,
                                               loss_mask)

        if get_sequence_parallel_world_size() > 1:
            policy_logps = gather_forward_split_backward(
                policy_logps,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')
            ref_logps = gather_forward_split_backward(
                ref_logps,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')

        if not self.use_varlen_attn:
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_logps(
                 policy_logps, ref_logps, all_loss_mask)
        else:
            message_hub = MessageHub.get_instance('varlen_attn_args')
            rank = dist.get_rank()
            cu_seqlens = message_hub.get_info(f'cumulative_len_rank_{rank}')
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_var_len_atten_logps(
                 policy_logps, ref_logps, all_loss_mask, cu_seqlens,
                 data['attention_mask'])

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        if self.loss_type == 'sigmoid':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) -
                    F.logsigmoid(-self.beta * logits) * self.label_smoothing)
        elif self.loss_type == 'robust':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) +
                    F.logsigmoid(-self.beta * logits) *
                    self.label_smoothing) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == 'hinge':
            loss = torch.relu(1 - self.beta * logits)
        elif self.loss_type == 'ipo':
            # eqn (17) of the paper where beta is the regularization
            # parameter for the IPO loss, denoted by tau in the paper.  # noqa
            loss = (logits - 1 / (2 * self.beta))**2
        elif self.loss_type == 'kto_pair':
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps -
                         reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps -
                           reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = \
                policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected)
            # is estimated using the rejected (chosen) half.  # noqa
            loss = torch.cat(
                (
                    1 - F.sigmoid(self.beta *
                                  (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta *
                                  (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == 'sppo_hard':
            # In the paper (https://arxiv.org/pdf/2405.00675),
            # SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation
            # is conducted outside of the trainer class.
            # The version described here is the hard probability version,
            # where P in Equation (4.7) of Algorithm 1 is set to 1 for
            # the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            loss = (a - 0.5 / self.beta)**2 + (b + 0.5 / self.beta)**2
        elif self.loss_type == 'nca_pair':
            chosen_rewards = (policy_chosen_logps -
                              reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps -
                                reference_rejected_logps) * self.beta
            loss = (-F.logsigmoid(chosen_rewards) -
                    0.5 * F.logsigmoid(-chosen_rewards) -
                    0.5 * F.logsigmoid(-rejected_rewards))
        else:
            raise ValueError(
                f'Unknown loss type: {self.loss_type}. Should be one of '
                "['sigmoid', 'hinge', 'ipo', 'kto_pair', "
                "'sppo_hard', 'nca_pair', 'robust']")
        # for logging
        chosen_rewards = self.beta * (
            policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (
            policy_rejected_logps - reference_rejected_logps)
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        loss_dict = {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_acc': reward_acc,
            'reward_margin': (chosen_rewards - rejected_rewards).mean(),
        }
        return loss_dict
