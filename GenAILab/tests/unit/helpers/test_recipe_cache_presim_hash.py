# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Cache base-hash correctness for the pre-sim split.

These assert the RecipeCache.compute_base_hash contract end-to-end: the pre-sim
component dict is folded into the base hash correctly, and -- critically -- a
recipe with NO pre-sim step produces the SAME base hash it did before pre-sim
support existed (no spurious cache invalidation for the common case).

recipe_cache imports onnx at module top, so this skips where onnx is
unavailable and runs in CI / on the cluster.
"""

import pytest

pytest.importorskip("onnx", reason="recipe_cache imports onnx at module top")

from GenAILab.bench.recipe_cache import RecipeCache  # noqa: E402
from GenAILab.bench.yaml_config_parser import ResolvedStep  # noqa: E402
from GenAILab.qai_hub_lm.schema import Recipe, split_recipe  # noqa: E402


class _FakePrecision:
    """Minimal stand-in: compute_base_hash only calls weight_identity()."""

    def __init__(self, ident="w4a16"):
        self._ident = ident

    def weight_identity(self):
        return self._ident


@pytest.fixture
def cache(tmp_path):
    # env_hash is read as an attribute; set it deterministically.
    c = RecipeCache.__new__(RecipeCache)
    c.env_hash = "test-env"
    return c


def _pre(raw):
    # split_recipe yields the flat pre-sim step dicts; lower them to the
    # ResolvedStep tuple that compute_base_hash consumes in production
    # (see yaml_config_parser.parse_document).
    pre, _ = split_recipe(Recipe.model_validate(raw))
    return tuple(
        ResolvedStep(
            name=step["name"],
            technique_cls=object,
            recipe_kwargs={
                k: v for k, v in step.items() if k not in ("name", "dataset")
            },
        )
        for step in pre
    )


COMMON = dict(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    precision_config=_FakePrecision(),
    model_kwargs={},
    framework="onnx",
    component="backbone",
)


def test_no_presim_hash_equals_legacy_no_spinquant(cache):
    # The base hash with an empty pre_sim must equal the hash computed with
    # pre_sim=None -- i.e. no "spinquant" key is added. This guarantees existing
    # cache entries for non-rotated runs stay valid.
    h_empty = cache.compute_base_hash(**COMMON, pre_sim=_pre([{"name": "Calibration"}]))
    h_none = cache.compute_base_hash(**COMMON, pre_sim=None)
    assert h_empty == h_none


def test_presim_changes_hash(cache):
    h_plain = cache.compute_base_hash(**COMMON, pre_sim=None)
    h_rot = cache.compute_base_hash(
        **COMMON,
        pre_sim=_pre(
            [{"name": "SpinQuant", "enable_r1": True}, {"name": "Calibration"}]
        ),
    )
    assert h_rot != h_plain


def test_different_rotations_differ(cache):
    h_r1 = cache.compute_base_hash(
        **COMMON,
        pre_sim=_pre(
            [{"name": "SpinQuant", "enable_r1": True}, {"name": "Calibration"}]
        ),
    )
    h_r1r2 = cache.compute_base_hash(
        **COMMON,
        pre_sim=_pre(
            [
                {"name": "SpinQuant", "enable_r1": True, "enable_r2": True},
                {"name": "Calibration"},
            ]
        ),
    )
    assert h_r1 != h_r1r2


def test_hash_independent_of_postsim_chain(cache):
    # base hash folds ONLY pre-sim; the post-sim steps extend the chain hash
    # separately, so they must not affect the BASE hash.
    h_a = cache.compute_base_hash(
        **COMMON,
        pre_sim=_pre(
            [{"name": "SpinQuant", "enable_r1": True}, {"name": "Calibration"}]
        ),
    )
    h_b = cache.compute_base_hash(
        **COMMON,
        pre_sim=_pre(
            [
                {"name": "SpinQuant", "enable_r1": True},
                {"name": "SeqMSE"},
                {"name": "Calibration"},
            ]
        ),
    )
    assert h_a == h_b
