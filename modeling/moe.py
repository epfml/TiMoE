"""
Simple MoE routing implementations that replace the MLP block in a standard transformer.
References:
1) Mistral Source for Mixtral MoEs: 
https://github.com/mistralai/mistral-src
2) ST-MoE:
https://arxiv.org/abs/2202.08906
3) Our notepad of MoE resources: 
https://docs.google.com/document/d/1NuQ5jr7V-Jv1ui7p4KrxO_JTz-7bpYcYMmh49EeJ-QA/edit?usp=sharing
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bisect
import math

class MoE(nn.Module):
    """
    Simplest MoE implementation with a linear router and softmax over experts.

    Note that in this implementation, we simply loop over the experts and
    aggregate the results. This is not the most efficient way to do it, but
    it also avoids the large memory overhead _and_ has no token dropping
    (because we do not need the capacity factor).
    """

    def __init__(self, config, mlp):
        super().__init__()
        assert config.moe_num_experts > 0
        self.experts = nn.ModuleList(
            [mlp(config=config) for _ in range(config.moe_num_experts)]
        )
        self.router = nn.Linear(config.n_embd, config.moe_num_experts, bias=False)
        self.top_k = config.moe_num_experts_per_tok
        self.softmax_order = config.moe_softmax_order

    def forward(self, inputs: torch.Tensor):
        # [batch_size * sequence_length, n_embd]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs_squashed)

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1)
            weights, selected_experts = torch.topk(all_probs, self.top_k)
        elif self.softmax_order == "topk_softmax":
            weights, selected_experts = torch.topk(router_logits, self.top_k)
            weights = F.softmax(weights, dim=-1)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        results = torch.zeros_like(inputs_squashed)
        # naive looping over experts
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            expert_input = inputs_squashed[batch_idx] #+ self.attn(inputs_squashed[batch_idx])
            output, _ = expert(expert_input)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * output.squeeze(0)

        # return results and router logits (for aux loss calculation later)
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
        }


class DummyExpert(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self._output_size = output_size
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((self._output_size,), device=inputs.device)
        return out, {}
    
    

class MaskedMoE(MoE):
    def __init__(self, config, mlp):
        super().__init__(config, mlp)
        self._sequence_length = config.sequence_length
        self.experts.append(DummyExpert(config.n_embd))
        self.router = nn.Linear(config.n_embd, config.moe_num_experts+1, bias=False)


    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        router_logits = self.router(inputs_squashed)
        mask = torch.cat(
            (mask, torch.ones((mask.shape[0], 1), device=mask.device)),
            dim=1
        )
        mask = mask.repeat_interleave(self._sequence_length, dim=0)
        router_logits = router_logits*mask

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1)
            weights, selected_experts = torch.topk(all_probs, self.top_k)
        elif self.softmax_order == "topk_softmax":
            weights, selected_experts = torch.topk(router_logits, self.top_k)
            weights = F.softmax(weights, dim=-1)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        results = torch.zeros_like(inputs_squashed)
        # naive looping over experts
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            expert_input = inputs_squashed[batch_idx]
            output, _ = expert(expert_input)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * output.squeeze(0)

        # return results and router logits (for aux loss calculation later)
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
        }
    

class TimeDependantMoE(nn.Module):
    def __init__(self, config, mlp):
        super().__init__()
        self._num_experts = config.moe_num_experts
        self._mask_moe = MaskedMoE(config, mlp)

    def forward(self, x, date):
        mask_date = torch.zeros(x.shape[0], self._num_experts).to(x.device)
        range_tensor = torch.arange(self._num_experts).unsqueeze(0).to(x.device)
        mask_date = (range_tensor < date.unsqueeze(1)).float()
        return self._mask_moe(x, mask_date)