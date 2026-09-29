"""Serializable model configuration; independent of cluster deployment."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class OpenJevConfig:
    backend_id: str = "qwen35"
    adapter_api_version: int = 1
    hidden_size: int = 2560
    head: dict = field(default_factory=lambda: {
        "set_dim": 512, "set_layers": 2, "set_heads": 8,
        "ffn_dim": 2048, "dropout": 0.0, "init_seed": 0, "score_init_gain": 0.1,
    })
    serializer: dict = field(default_factory=lambda: {
        "version": 1, "add_bos_token": False,
        "max_branch_length": 32768, "max_decision_tokens": 32768,
    })

    compute_contract: str = "legacy-v0"
    kernel_backend: str = "default"

    def __post_init__(self):
        from openjev.compute import validate_compute_contract
        validate_compute_contract(self.compute_contract)
        if self.kernel_backend not in {"default", "fla"} or (self.kernel_backend == "fla" and self.backend_id != "qwen35"):
            raise ValueError("unsupported kernel backend for this model family")
        if self.adapter_api_version != 1 or self.hidden_size < 1:
            raise ValueError("unsupported adapter API or invalid hidden size")
        if self.serializer.get("version") != 1:
            raise ValueError("unsupported serializer version")

    def to_dict(self):
        return asdict(self)

    @property
    def tie_word_embeddings(self):
        # Decision heads have no tied vocabulary-output projection.
        return False

    def save_pretrained(self, directory):
        from pathlib import Path
        from openjev.artifacts import write_json
        write_json(Path(directory) / "openjev_config.json", self.to_dict())

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
