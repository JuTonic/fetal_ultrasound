"""Model/contract checks for the pinned Docker environment; no weight downloads."""
import pytest
import torch

from fetal.data import CLASSES9
from fetal.models import TRANSFORM, WEIGHTS, build_model, load_checkpoint


@pytest.mark.parametrize("arch", list(WEIGHTS))
def test_all_architectures_produce_nine_logits(arch):
    torch.set_num_threads(1)
    model = build_model(arch).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 9)
    assert torch.isfinite(output).all()


def test_checkpoint_roundtrip_and_rejection_of_wrong_class_order(tmp_path):
    torch.set_num_threads(1)
    model = build_model("resnet18").eval()
    contract = {
        "arch": "resnet18", "classes": CLASSES9, "transform": TRANSFORM,
        "config": {}, "seed": 0, "selected_epoch": 1,
        "initialization": "scratch", "training_groups": [1],
    }
    path = tmp_path / "model.pt"
    torch.save({"contract": contract, "state_dict": model.state_dict()}, path)
    restored, _ = load_checkpoint(path, {"classes": CLASSES9, "transform": TRANSFORM})
    restored.eval()
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))
    with pytest.raises(ValueError, match="classes"):
        load_checkpoint(path, {"classes": list(reversed(CLASSES9))})
    with pytest.raises(ValueError, match="transform"):
        load_checkpoint(path, {"transform": "wrong"})
