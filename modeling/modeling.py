from transformers import PreTrainedModel
from .configuration import TiMoEConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.utils import ModelOutput
from .gpt import GPTBase
from .aux_losses import entropy_reg, load_balancing_loss, router_z_loss
from typing import Optional, List
from dataclasses import dataclass
import tiktoken


@dataclass
class Output(ModelOutput):
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    expert_losses: Optional[List] = None
    loss_to_log: Optional[float] = None
    router_logits: Optional[torch.FloatTensor] = None     
    selected_experts: Optional[torch.LongTensor] = None
    combined_log_probs: Optional[torch.FloatTensor] = None


class TiMoE(PreTrainedModel):
    config_class = TiMoEConfig

    def __init__(self, config, expert_weights=None, dropout=0.1):
        """
        Constructor for the TiMoE (Mixture of Language Models) class.

        :param config: The configuration of the model (should be a PretrainedConfig object)
        :param expert_weights: (Optional) A list of weights for each expert to load pre-trained weights (should match the number of experts)
        :param dropout: Dropout rate for the model
        :param use_router: Flag to indicate whether to use routing (currently not implemented)
        """
        super(TiMoE, self).__init__(config)
        
        # Number of experts
        self.num_experts = config.num_experts
        # print(f"Number of experts: {self.num_experts}")
        # print(f"Expert configurations: {config.expert_configs}")
        assert len(config.expert_configs) == self.num_experts, "Number of expert configurations must match num_experts in config."
        self.expert_configs = config.expert_configs

        
        self.use_router = config.use_router
        
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, self.num_experts),
        )
        self.top_k = config.top_k_experts if hasattr(config, "top_k_experts") else self.num_experts

        # Initialize experts using the provided configurations
        self.experts = nn.ModuleList([GPTBase(config=self.expert_configs[i]) for i in range(self.num_experts)])
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Load pre-trained weights if provided
        if expert_weights is not None:
            for i, expert in enumerate(self.experts):
                expert.load_state_dict(expert_weights[i], strict=False)
                expert.transformer.wte.weight = torch.nn.Parameter(expert.transformer.wte.weight.clone())
                for param in expert.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, targets=None, date=None, masking_enabled=True, **kwargs):
        """
        Forward pass for the TiMoE model, passing input through all experts and averaging their outputs.

        :param input_ids: Input token IDs (batch_size, seq_len)
        :param attention_mask: Attention mask (batch_size, seq_len)
        :param targets: Target labels for calculating loss (batch_size, seq_len)
        :param date: A tensor indicating which experts to use. Each sample in the batch can have a different date.
        :param masking_enabled: Whether or not to perform expert masking (True/False)
        :param kwargs: Additional arguments
        :return: The averaged output of all active experts up to the specified date for each sample in the batch
        """
        device = input_ids.device
        b, t = input_ids.size()

        # Ensure the sequence length doesn't exceed the configured block size
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"

        # If date is None, set a default value (e.g., 6 for all samples)
        if date is None:
            date = torch.full((1, b), 6, dtype=torch.long, device=device).squeeze(0)
        elif isinstance(date, int):
            # If date is an integer, set it for all samples in the batch
            date = (date - 2013) // 2 + 1
            date = torch.full((1, b), date, dtype=torch.long, device=device).squeeze(0)
        elif isinstance(date, torch.Tensor):
            # Ensure the tensor has the correct shape (batch_size,)
            assert date.size(0) == b, "The size of date tensor must match the batch size."
            date = date.to(device)

        # Get outputs from each expert
        expert_outputs = []
        expert_losses = []

        # Track the number of active experts for each sample in the batch
        active_experts_count = torch.zeros(b, dtype=torch.long, device=device)

        # Pass input through each expert
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                # Masking logic based on date (for each sample in the batch)
                expert_mask = date >= i  # Mask experts where date < i (i.e., deactivate them)
                #expert_mask = date <= i 
                # Expand the expert_mask to match the logits shape (batch_size, 1, 1)
                expert_mask_expanded = expert_mask.unsqueeze(-1).unsqueeze(-1).float()

                expert_output = expert(input_ids, targets=targets, date=date, **kwargs, get_logits=True)

                logits = expert_output["logits"]
                loss_to_log = expert_output["loss_to_log"]

                # Mask out the outputs for deactivated experts
                logits = logits * expert_mask_expanded  # Apply the mask (zero out logits for inactive experts)

                # Only append logits from active experts
                expert_outputs.append(logits)
                expert_losses.append(loss_to_log)

                # Update active expert count for each sample
                active_experts_count += expert_mask.long()  # Ensure type consistency by converting `expert_mask` to Long

        # Stack the logits and calculate the mean for each sample across the active experts
        expert_outputs = torch.stack(expert_outputs, dim=0)  # Shape: (num_experts, batch_size, seq_len, vocab_size)

        # Convert logits to log-probabilities for each expert
        log_probs = F.log_softmax(expert_outputs, dim=-1)
        
        if self.use_router:
            hidden = self.experts[0].transformer.wte(input_ids)  # (B, T, D)
            pooled_hidden = hidden.mean(dim=1)  # (B, D)
            router_logits = self.router(pooled_hidden)  # (B, E)

            expert_ids = torch.arange(self.num_experts, device=input_ids.device)
            router_mask = date.unsqueeze(1) >= expert_ids.unsqueeze(0)  # (B, E)
            masked_router_logits = router_logits.masked_fill(~router_mask, float("-inf"))

            # Select top-k
            topk_probs, topk_indices = torch.topk(F.softmax(masked_router_logits, dim=-1), self.top_k, dim=-1)
            sparse_probs = torch.zeros_like(router_logits)
            sparse_probs.scatter_(1, topk_indices, topk_probs)
            sparse_probs = sparse_probs / sparse_probs.sum(dim=1, keepdim=True)

            # Convert weights to log-space
            log_weights = torch.log(sparse_probs + 1e-9)  # (B, E)

            # Broadcast for logsumexp: (E, B, T, V)
            log_weights_exp = log_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # (E, B, 1, 1)
            weighted_log_probs = log_probs + log_weights_exp  # (E, B, T, V)

            combined_log_probs = torch.logsumexp(weighted_log_probs, dim=0)  # (B, T, V)

        else:
            # Unweighted average in log-prob space across active experts (equal weights)
            log_weights = torch.log(1.0 / active_experts_count.float().clamp(min=1.0)).view(1, -1, 1, 1)  # (1, B, 1, 1)
            weighted_log_probs = log_probs + log_weights
            combined_log_probs = torch.logsumexp(weighted_log_probs, dim=0)  # (B, T, V)

        # Calculate the loss if targets are provided
        if targets is not None:
            loss = F.nll_loss(combined_log_probs.view(-1, combined_log_probs.size(-1)), targets.view(-1), ignore_index=-1)
            loss_to_log = loss.item()

            # Add auxiliary router losses (only if routing is used and we're training)
            if self.use_router and self.training:
                flat_router_logits = router_logits.view(-1, router_logits.size(-1))  # (B*T, E)
                flat_selected_experts = topk_indices.view(-1, topk_indices.size(-1))  # (B*T, top_k)

                # Compute each auxiliary loss
                entropy = entropy_reg(flat_router_logits)
                lb_loss = load_balancing_loss(flat_router_logits, flat_selected_experts)
                zloss = router_z_loss(flat_router_logits)

                # Combine them with your preferred weights
                loss = (
                    loss
                    + 0.01 *entropy
                    + 0.01 * lb_loss
                    + 0.0001 * zloss
                )
        else:
            loss = None
            loss_to_log = None

        return Output(
            logits=expert_outputs,
            loss=loss,
            combined_log_probs=combined_log_probs,
            loss_to_log=loss_to_log,
            expert_losses=expert_losses,
            router_logits=router_logits if self.use_router else None,
            selected_experts=topk_indices if self.use_router else None,
        )


    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, date=None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = input_ids
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = (
                idx
                if idx.size(1) <= self.config.sequence_length
                else idx[:, -self.config.sequence_length :]
            )
            # # forward the model to get the logits for the index in the sequence
            # logits = self(idx_cond, date, get_logits=True).logits
            # # pluck the logits at the final step and scale by desired temperature
            # logits = logits[:, -1, :] / temperature
            # # optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float("Inf")
            # # apply softmax to convert logits to (normalized) probabilities
            # probs = F.softmax(logits, dim=-1)
            # # sample from the distribution
            log_probs = self(idx_cond, date=date).combined_log_probs[:, -1, :]
            #idx_next = torch.multinomial(probs, num_samples=1)
            # Sample from the log probabilities
            if temperature == 0:
                # If temperature is 0, take the argmax (greedy sampling)
                idx_next = torch.argmax(log_probs, dim=-1, keepdim=True)
            else:
                # Apply temperature scaling
                scaled_log_probs = log_probs / temperature
                # Convert log probabilities to probabilities
                probs = torch.exp(scaled_log_probs)
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # check if we hit the end of the sequence
            if idx_next.item() == 50526:
                break

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, date=None, temperature=1.0, top_k=None):
        idx = (
            torch.tensor(
                self.tokenizer.encode(in_str)
            )
            .view(1, -1)
            .to(next(self.parameters()).device)
        )
        out_idx = (
            self.generate(idx, max_new_tokens, date, temperature, top_k)
            .view(-1)
            .to("cpu")
            .numpy()
        )
        return self.tokenizer.decode(out_idx)
    
