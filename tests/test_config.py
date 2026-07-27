from seminovo.config import Config


def test_public_config_exposes_optional_checkpoint():
    config = Config("configs/seminovo.yaml")

    assert config.load_checkpoint is None
    assert config.dim_model == 512
    assert config.gated_attention is True
    assert config.max_epochs == 30
