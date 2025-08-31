from transformers import PretrainedConfig

class TiMoEConfig(PretrainedConfig):
    model_type = "TiMoE"

    def __init__(
        self,
        vocab_size=50304,
        n_embd=768,
        n_layer=12,
        n_head=12,
        sequence_length=1024,
        mlp_dim_exp_factor=1.0,
        dropout=0.0,
        bias=False,
        num_experts=6,
        expert_configs=None,
        use_router=False,
        top_k_experts=6,
        architectures=["TiMoE"],
        auto_map={
            "AutoConfig": "configuration.TiMoEConfig",
            "AutoModelForCausalLM": "modeling.TiMoE",
            "AutoTokenizer": "GPT2TokenizerFast"
        },
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.sequence_length = sequence_length
        self.mlp_dim_exp_factor = mlp_dim_exp_factor
        self.dropout = dropout
        self.bias = bias
        self.num_experts = num_experts
        self.expert_configs = expert_configs
        self.use_router = use_router
        self.architectures = architectures
        self.auto_map = auto_map
        self.top_k_experts = top_k_experts
    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.update({
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "sequence_length": self.sequence_length,
            "mlp_dim_exp_factor": self.mlp_dim_exp_factor,
            "dropout": self.dropout,
            "bias": self.bias,
            "num_experts": self.num_experts,
            "expert_configs": [
                expert_config.to_dict() if not isinstance(expert_config, dict) else expert_config 
                for expert_config in self.expert_configs
            ] if self.expert_configs else None,  
            "use_router": self.use_router,
            "top_k_experts": self.top_k_experts,
        })
        return config_dict