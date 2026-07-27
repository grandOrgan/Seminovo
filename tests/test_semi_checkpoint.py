import torch

from seminovo.ema import ExponentialMovingAverage
from seminovo.train_semi import initialize_from_checkpoint


def test_checkpoint_ema_initializes_student_and_teacher(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    model.ema = ExponentialMovingAverage(0.999)
    checkpoint = {
        "state_dict": {"weight": torch.tensor([[1.0]])},
        "ema_state_dict": {"weight": torch.tensor([[4.0]])},
    }
    path = tmp_path / "best.ckpt"
    torch.save(checkpoint, path)

    initialize_from_checkpoint(model, path)

    torch.testing.assert_close(model.weight, torch.tensor([[4.0]]))
    torch.testing.assert_close(model.ema.shadow["weight"], torch.tensor([[4.0]]))
