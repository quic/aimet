# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pre-sim / on-sim split and the cache-identity contract.

These are the safety net for the dedicated pre-sim pass: they pin the exact
behavior the runner + cache depend on, especially the cache-hash rules that, if
wrong, would silently load the wrong cached weights. Pure pydantic -- no aimet.
"""

import pytest

from GenAILab.qai_hub_lm.schema import (
    Recipe,
    has_pre_sim,
    pre_sim_flags,
    pre_sim_identity,
    split_recipe,
)


def _r(raw):
    return Recipe.model_validate(raw)


class TestSplit:
    # split_recipe returns (pre_sim: FLAT list, on_sim: {component: list}).
    def test_spinquant_goes_to_presim_rest_to_on_sim(self):
        pre, on_sim = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "AdaScale", "num_iterations": 64},
                    {"name": "Calibration"},
                ]
            )
        )
        assert [s["name"] for s in pre] == ["SpinQuant"]
        assert [s["name"] for s in on_sim["backbone"]] == ["AdaScale", "Calibration"]

    def test_no_presim_yields_empty_presim_list(self):
        pre, on_sim = split_recipe(_r([{"name": "Calibration"}]))
        assert pre == []
        assert [s["name"] for s in on_sim["backbone"]] == ["Calibration"]

    def test_vlm_presim_collapsed_to_one_flat_list(self):
        # identical prefixes across components collapse to ONE pre_sim list;
        # on_sim stays per-component.
        pre, on_sim = split_recipe(
            _r(
                {
                    "backbone": [
                        {
                            "name": "SpinQuant",
                            "enable_r1": True,
                            "enable_r2": False,
                            "enable_r3": False,
                        },
                        {"name": "Calibration"},
                    ],
                    "visual": [
                        {
                            "name": "SpinQuant",
                            "enable_r1": True,
                            "enable_r2": False,
                            "enable_r3": False,
                        },
                        {"name": "Calibration"},
                    ],
                }
            )
        )
        assert [s["name"] for s in pre] == ["SpinQuant"]
        assert [s["name"] for s in on_sim["backbone"]] == ["Calibration"]
        assert [s["name"] for s in on_sim["visual"]] == ["Calibration"]


class TestHasPreSim:
    def test_true_with_spinquant(self):
        pre, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )
        )
        assert has_pre_sim(pre) is True

    def test_false_without(self):
        pre, _ = split_recipe(_r([{"name": "Calibration"}]))
        assert has_pre_sim(pre) is False


class TestPreSimFlags:
    def test_extracts_flat_flags_for_backend(self):
        pre, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": True,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )
        )
        # All three flags are always present: they are contract-filled at
        # validation, so an omitted flag lowers explicitly rather than silently.
        assert pre_sim_flags(pre, "SpinQuant") == {
            "enable_r1": True,
            "enable_r2": True,
            "enable_r3": False,
        }

    def test_returns_none_when_absent(self):
        pre, _ = split_recipe(_r([{"name": "Calibration"}]))
        assert pre_sim_flags(pre, "SpinQuant") is None

    def test_divergent_component_prefixes_rejected(self):
        # Strict: components can't have differing pre-sim flags (this replaces the
        # old silent "prefer backbone" behavior with a loud validation error).
        with pytest.raises(Exception):
            _r(
                {
                    "backbone": [
                        {
                            "name": "SpinQuant",
                            "enable_r1": True,
                            "enable_r2": True,
                            "enable_r3": False,
                        },
                        {"name": "Calibration"},
                    ],
                    "visual": [
                        {
                            "name": "SpinQuant",
                            "enable_r1": True,
                            "enable_r2": False,
                            "enable_r3": False,
                        },
                        {"name": "Calibration"},
                    ],
                }
            )


class TestCacheIdentity:
    """The contract the cache base hash depends on. Getting these wrong silently
    loads the wrong cached weights, so each is pinned explicitly."""

    def test_none_when_no_presim_so_hash_key_omitted(self):
        # CRITICAL: a non-rotated run must add NO pre-sim key to the hash, so its
        # cache key is byte-identical to the pre-pre-sim behavior.
        pre, _ = split_recipe(_r([{"name": "Calibration"}]))
        assert pre_sim_identity(pre) is None

    def test_identity_present_when_spinquant(self):
        pre, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )
        )
        assert pre_sim_identity(pre) == {
            "SpinQuant": {"enable_r1": True, "enable_r2": False, "enable_r3": False}
        }

    def test_different_rotation_flags_give_different_identity(self):
        # rotated-R1-only vs rotated-R1+R2 must NOT collide on the cache key.
        pre_a, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )
        )
        pre_b, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": True,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )
        )
        assert pre_sim_identity(pre_a) != pre_sim_identity(pre_b)

    def test_identity_independent_of_postsim_steps(self):
        # The base hash folds ONLY pre-sim; post-sim steps extend the chain hash
        # separately. So identity must be the same regardless of the post chain.
        pre_a, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )
        )
        pre_b, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "SeqMSE"},
                    {"name": "Calibration"},
                ]
            )
        )
        assert pre_sim_identity(pre_a) == pre_sim_identity(pre_b)

    def test_identity_independent_of_dataset(self):
        # dataset is irrelevant to a pre-sim rotation identity.
        pre_a, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {"name": "Calibration"},
                ]
            )
        )
        pre_b, _ = split_recipe(
            _r(
                [
                    {
                        "name": "SpinQuant",
                        "enable_r1": True,
                        "enable_r2": False,
                        "enable_r3": False,
                    },
                    {
                        "name": "Calibration",
                        "dataset": {"name": "C4", "split": "train"},
                    },
                ]
            )
        )
        assert pre_sim_identity(pre_a) == pre_sim_identity(pre_b)
