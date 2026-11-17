import copy
import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.common import ROOT, config
from src.data import ClinicalEncoder, assign_split, group_components
from src.models import FusionModel
from vendor.healnet.healnet import HealNet


torch.set_num_threads(2)


def test_duplicate_connected_patients_share_partition():
    rows = pd.DataFrame({"patient_id": ["A", "B", "B", "C"], "img_id": ["1", "2", "3", "4"]})
    info = [{"rgb_sha256": h} for h in ["same", "same", "other", "third"]]
    groups = group_components(rows, info)
    assert groups[0] == groups[1] == groups[2]
    assert assign_split(groups[0], config()) == assign_split(groups[2], config())


def test_ambiguous_lesion_ids_cannot_cross_partitions():
    rows = pd.DataFrame({'patient_id': ['A', 'B', 'B', 'C'], 'lesion_id': ['1', '1', '2', '3'], 'img_id': ['a', 'b', 'c', 'd']})
    groups = group_components(rows, [{'rgb_sha256': str(i)} for i in range(4)])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_future_data_cannot_change_training_preprocessing():
    train = pd.DataFrame({"age": [20, 40, None], "region": ["arm", "leg", None]})
    encoder = ClinicalEncoder(["age"], ["region"]).fit(train)
    frozen = copy.deepcopy(encoder.state)
    transformed = encoder.transform(pd.DataFrame({"age": [1000, None], "region": ["unseen", None]}))
    assert encoder.state == frozen
    assert transformed[0, -1] == 1
    assert transformed[1, 1] == 1
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("name", config()["models"])
def test_real_model_forward_backward_and_optimizer(name):
    torch.manual_seed(17)
    model = FusionModel(name, 23, config()).train()
    image = torch.randn(4, 7, 7, 512, requires_grad=True)
    clinical = torch.randn(4, 23, requires_grad=True)
    output = model(image, clinical)
    assert output.shape == (4, 6)
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 2, 3])); loss.backward()
    assert torch.isfinite(loss)
    if name in ("metablock", "healnet", "concat_linear"):
        assert image.grad.abs().sum() > 0
        assert clinical.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.optim.AdamW(model.parameters()).step()


def tiny_healnet():
    return HealNet(n_modalities=2, channel_dims=[3, 4], num_spatial_axes=[1, 2], out_dims=6, depth=1, l_c=4, l_d=8, x_heads=1, l_heads=1, cross_dim_head=4, latent_dim_head=4, self_per_cross_attn=1)


def test_healnet_preserves_inputs_and_fails_on_bad_channels():
    model = tiny_healnet().eval(); tensors = [torch.randn(2, 1, 3), torch.randn(2, 3, 3, 4)]
    originals = [x.clone() for x in tensors]
    model(tensors)
    assert all(torch.equal(x, y) for x, y in zip(tensors, originals))
    with pytest.raises(RuntimeError):
        model([torch.randn(2, 1, 7), tensors[1]])


def test_label_and_postdiagnosis_columns_excluded():
    c = config(); selected = set(c["numeric_features"] + c["categorical_features"])
    assert not selected & {"diagnostic", "biopsed", "patient_id", "lesion_id", "img_id"}
    assert len(selected) == 14


def test_hardened_healnet_matches_original_on_valid_inputs():
    from vendor.healnet.upstream import HealNet as Original
    args = dict(n_modalities=2, channel_dims=[3, 4], num_spatial_axes=[1, 2], out_dims=6, depth=1, l_c=4, l_d=8, x_heads=1, l_heads=1, cross_dim_head=4, latent_dim_head=4, self_per_cross_attn=1)
    original = Original(**args).eval(); hardened = tiny_healnet().eval()
    hardened.load_state_dict(original.state_dict())
    inputs = [torch.randn(2, 1, 3), torch.randn(2, 3, 3, 4)]
    with torch.inference_mode():
        expected = original([x.clone() for x in inputs]); actual = hardened(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_balanced_batches_keep_every_row_without_singletons():
    from src.run import batches
    for n in (64, 65, 129, 1001):
        parts = batches(np.arange(n), 64)
        np.testing.assert_array_equal(np.concatenate(parts), np.arange(n))
        assert min(map(len, parts)) >= 2


def test_label_access_rejects_holdout_before_reading(tmp_path):
    from src.run import Store
    store = object.__new__(Store); store.allowed = {'fit', 'tune'}; store.reads = {}
    with pytest.raises(ValueError, match='Forbidden'):
        store.labels('test')


def test_event_chain_rejects_tampering_and_post_terminal(tmp_path):
    from src.run import Journal
    from src.supervisor import validate_journal
    (tmp_path / 'start.json').write_text('{}')
    journal = Journal(tmp_path); journal.emit('run_start'); journal.emit('terminal', status='success')
    assert len(validate_journal(tmp_path)) == 2
    with pytest.raises(ValueError, match='post-terminal'):
        journal.emit('checkpoint')
    path = tmp_path / 'events.jsonl'; path.write_text(path.read_text().replace('run_start', 'checkpoint'))
    with pytest.raises(ValueError, match='event chain'):
        validate_journal(tmp_path)


def test_missing_classes_are_visible_in_metrics():
    from src.metrics import score
    probability = np.array([[.9, .02, .02, .02, .02, .02]])
    result = score(np.array([0]), probability)
    assert result['macro_auroc'] is None
    assert result['class_support'] == [1, 0, 0, 0, 0, 0]
    assert result['macro_f1'] == pytest.approx(1 / 6)
