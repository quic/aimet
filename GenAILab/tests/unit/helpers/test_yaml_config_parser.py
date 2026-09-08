# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for YAMLConfigParser registry and config parsing."""

import copy
from unittest.mock import patch, MagicMock

import pytest

from GenAILab.bench.yaml_config_parser import (
    YAMLConfigParser,
    AdaptationInfo,
    ModelConfig,
)


# ---------------------------------------------------------------------------
# Fixtures: save and restore global registry state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the parser's global lookup dicts around each test."""
    saved = {
        "recipe": dict(YAMLConfigParser.recipe_lookup),
        "model": dict(YAMLConfigParser.model_lookup),
        "dataset": dict(YAMLConfigParser.dataset_lookup),
        "metrics": dict(YAMLConfigParser.metrics_lookup),
        "adaptation": dict(YAMLConfigParser.adaptation_lookup),
        "default_llm": YAMLConfigParser._default_llm_cls,
    }
    yield
    YAMLConfigParser.recipe_lookup = saved["recipe"]
    YAMLConfigParser.model_lookup = saved["model"]
    YAMLConfigParser.dataset_lookup = saved["dataset"]
    YAMLConfigParser.metrics_lookup = saved["metrics"]
    YAMLConfigParser.adaptation_lookup = saved["adaptation"]
    YAMLConfigParser._default_llm_cls = saved["default_llm"]


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_metric(self):
        @YAMLConfigParser.register_metric
        class MyMetric:
            pass

        assert YAMLConfigParser.metrics_lookup["MyMetric"] is MyMetric

    def test_register_dataset(self):
        from GenAILab.qai_hub_lm.schema.dataset import WikitextSpec

        @YAMLConfigParser.register_dataset(WikitextSpec)
        class MyDataset:
            pass

        # register_dataset keys the lookup by the spec's wire name ("Wikitext").
        assert YAMLConfigParser.dataset_lookup["Wikitext"] is MyDataset

    def test_register_recipe(self):
        from GenAILab.qai_hub_lm.schema.recipe import RemoveQuantizationSpec

        # register_recipe binds the lowering to its spec and enforces that
        # apply()'s explicit kwargs match the spec exactly (fixtures excluded).
        @YAMLConfigParser.register_recipe(RemoveQuantizationSpec)
        class MyRecipe:
            @staticmethod
            def apply(quantsim, generator, dataloader, **kwargs):
                pass

        # keyed by the spec's wire name ("RemoveQuantization").
        assert YAMLConfigParser.recipe_lookup["RemoveQuantization"] is MyRecipe

    def test_register_model(self):
        @YAMLConfigParser.register_model("test_vlm")
        class MyVLM:
            pass

        assert YAMLConfigParser.model_lookup["test_vlm"] is MyVLM

    def test_register_model_duplicate_raises(self):
        @YAMLConfigParser.register_model("dup_type")
        class First:
            pass

        with pytest.raises(RuntimeError, match="already registered"):

            @YAMLConfigParser.register_model("dup_type")
            class Second:
                pass

    def test_register_adaptation(self):
        @YAMLConfigParser.register_adaptation("SHA", model_type="llama")
        class SHAMixin:
            pass

        key = ("llama", "SHA")
        assert key in YAMLConfigParser.adaptation_lookup
        assert YAMLConfigParser.adaptation_lookup[key].mixin_cls is SHAMixin

    def test_register_adaptation_exclusive(self):
        @YAMLConfigParser.register_adaptation("AIHM", model_type="*", exclusive=True)
        class AIHMMixin:
            pass

        key = ("*", "AIHM")
        assert YAMLConfigParser.adaptation_lookup[key].exclusive is True

    def test_register_adaptation_required_for_export(self):
        @YAMLConfigParser.register_adaptation(
            "ExportHelper", model_type="llama", required_for_export=True
        )
        class ExportHelperMixin:
            pass

        key = ("llama", "ExportHelper")
        info = YAMLConfigParser.adaptation_lookup[key]
        assert info.required_for_export is True
        assert info.exclusive is False

    def test_register_default_llm(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = None
        YAMLConfigParser.register_default_llm(FakeLLM)
        assert YAMLConfigParser._default_llm_cls is FakeLLM

    def test_register_default_llm_duplicate_different_raises(self):
        class LLM_A:
            pass

        class LLM_B:
            pass

        YAMLConfigParser._default_llm_cls = None
        YAMLConfigParser.register_default_llm(LLM_A)
        with pytest.raises(RuntimeError, match="already registered"):
            YAMLConfigParser.register_default_llm(LLM_B)

    def test_register_default_llm_same_class_ok(self):
        class LLM_A:
            pass

        YAMLConfigParser._default_llm_cls = None
        YAMLConfigParser.register_default_llm(LLM_A)
        YAMLConfigParser.register_default_llm(LLM_A)  # no error

    def test_get_default_llm_unregistered_raises(self):
        YAMLConfigParser._default_llm_cls = None
        with pytest.raises(RuntimeError, match="No default LLM"):
            YAMLConfigParser.get_default_llm()


# ---------------------------------------------------------------------------
# Adaptation resolution
# ---------------------------------------------------------------------------


class TestAdaptationResolution:
    def test_get_model_class_no_adaptations(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM
        result = YAMLConfigParser.get_model_class("llama")
        assert result is FakeLLM

    def test_get_model_class_registered_model(self):
        @YAMLConfigParser.register_model("qwen2_vl")
        class Qwen2VL:
            pass

        result = YAMLConfigParser.get_model_class("qwen2_vl")
        assert result is Qwen2VL

    def test_get_model_class_with_adaptation(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM

        @YAMLConfigParser.register_adaptation("SHA", model_type="llama")
        class SHAMixin:
            pass

        result = YAMLConfigParser.get_model_class("llama", adaptations=["SHA"])
        # Should be a dynamically created class mixing SHAMixin + FakeLLM
        assert issubclass(result, FakeLLM)
        assert issubclass(result, SHAMixin)

    def test_get_model_class_universal_adaptation(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM

        @YAMLConfigParser.register_adaptation("FastExport", model_type="*")
        class FastExportMixin:
            pass

        result = YAMLConfigParser.get_model_class(
            "any_type", adaptations=["FastExport"]
        )
        assert issubclass(result, FastExportMixin)

    def test_exclusive_adaptation_alone_ok(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM

        @YAMLConfigParser.register_adaptation("AIHM", exclusive=True)
        class AIHMMixin:
            pass

        result = YAMLConfigParser.get_model_class("llama", adaptations=["AIHM"])
        assert issubclass(result, AIHMMixin)

    def test_exclusive_adaptation_combined_raises(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM

        @YAMLConfigParser.register_adaptation("AIHM", exclusive=True)
        class AIHMMixin:
            pass

        @YAMLConfigParser.register_adaptation("SHA")
        class SHAMixin:
            pass

        with pytest.raises(ValueError, match="exclusive"):
            YAMLConfigParser.get_model_class("llama", adaptations=["AIHM", "SHA"])

    def test_unknown_adaptation_raises(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM
        with pytest.raises(LookupError, match="No 'Nonexistent'"):
            YAMLConfigParser.get_model_class("llama", adaptations=["Nonexistent"])

    def test_get_required_export_adaptations(self):
        @YAMLConfigParser.register_adaptation(
            "ReqA", model_type="llama", required_for_export=True
        )
        class ReqAMixin:
            pass

        @YAMLConfigParser.register_adaptation(
            "OptB", model_type="llama", required_for_export=False
        )
        class OptBMixin:
            pass

        @YAMLConfigParser.register_adaptation(
            "ReqOther", model_type="qwen2", required_for_export=True
        )
        class ReqOtherMixin:
            pass

        result = YAMLConfigParser.get_required_export_adaptations("llama")
        assert "ReqA" in result
        assert "OptB" not in result
        assert "ReqOther" not in result

    def test_get_required_export_adaptations_excludes_exclusive(self):
        @YAMLConfigParser.register_adaptation(
            "ExclReq",
            model_type="llama",
            exclusive=True,
            required_for_export=True,
        )
        class ExclReqMixin:
            pass

        result = YAMLConfigParser.get_required_export_adaptations("llama")
        assert "ExclReq" not in result

    def test_adaptation_kwargs_set_as_class_attrs(self):
        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM

        @YAMLConfigParser.register_adaptation("Scale", model_type="llama")
        class ScaleMixin:
            layer_multipliers: dict = {}

        result_cls = YAMLConfigParser.get_model_class(
            "llama",
            adaptations=["Scale"],
            adaptation_kwargs={"Scale": {"layer_multipliers": {0: 2.0}}},
        )
        assert result_cls.layer_multipliers == {0: 2.0}


# ---------------------------------------------------------------------------
# Normalize adaptations
# ---------------------------------------------------------------------------


class TestNormalizeAdaptations:
    def test_string_entries(self):
        names, kwargs = YAMLConfigParser._normalize_adaptations(["SHA", "FastExport"])
        assert names == ["SHA", "FastExport"]
        assert kwargs == {}

    def test_dict_entries(self):
        raw = [{"AttentionMaskScale": {"layer_multipliers": {0: 10.0}}}]
        names, kwargs = YAMLConfigParser._normalize_adaptations(raw)
        assert names == ["AttentionMaskScale"]
        assert kwargs["AttentionMaskScale"] == {"layer_multipliers": {0: 10.0}}

    def test_mixed_entries(self):
        raw = ["SHA", {"AttentionMaskScale": {"layer_multipliers": {0: 2.0}}}]
        names, kwargs = YAMLConfigParser._normalize_adaptations(raw)
        assert names == ["SHA", "AttentionMaskScale"]
        assert "SHA" not in kwargs
        assert kwargs["AttentionMaskScale"] == {"layer_multipliers": {0: 2.0}}

    def test_dict_with_none_value(self):
        raw = [{"NoArgs": None}]
        names, kwargs = YAMLConfigParser._normalize_adaptations(raw)
        assert names == ["NoArgs"]
        assert kwargs["NoArgs"] == {}

    def test_multi_key_dict_raises(self):
        raw = [{"A": {}, "B": {}}]
        with pytest.raises(ValueError, match="exactly one key"):
            YAMLConfigParser._normalize_adaptations(raw)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="string or single-key dict"):
            YAMLConfigParser._normalize_adaptations([42])


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_missing_model_raises(self):
        with pytest.raises(RuntimeError, match="Model section"):
            YAMLConfigParser.validate_config({"metrics": [{"name": "PPL"}]})

    def test_missing_metrics_raises(self):
        with pytest.raises(RuntimeError, match="Metrics not"):
            YAMLConfigParser.validate_config(
                {
                    "model": {
                        "model_id": "x",
                        "sequence_length": 32,
                        "context_length": 64,
                    }
                }
            )

    def test_missing_model_id_raises(self):
        with pytest.raises(RuntimeError, match="model_id"):
            YAMLConfigParser.validate_config(
                {
                    "model": {"sequence_length": 32, "context_length": 64},
                    "metrics": [{"name": "PPL"}],
                }
            )

    def test_missing_sequence_length_raises(self):
        with pytest.raises(RuntimeError, match="Sequence length"):
            YAMLConfigParser.validate_config(
                {
                    "model": {"model_id": "x", "context_length": 64},
                    "metrics": [{"name": "PPL"}],
                }
            )

    def test_missing_context_length_raises(self):
        with pytest.raises(RuntimeError, match="Context length"):
            YAMLConfigParser.validate_config(
                {
                    "model": {"model_id": "x", "sequence_length": 32},
                    "metrics": [{"name": "PPL"}],
                }
            )

    def test_metric_missing_name_raises(self):
        with pytest.raises(RuntimeError, match="Metric name"):
            YAMLConfigParser.validate_config(
                {
                    "model": {
                        "model_id": "x",
                        "sequence_length": 32,
                        "context_length": 64,
                    },
                    "metrics": [{"class": "PPL"}],
                }
            )

    def test_recipe_name_normalized_to_backbone(self):
        doc = {
            "model": {"model_id": "x", "sequence_length": 32, "context_length": 64},
            "recipe": {"name": "Calibration"},
            "metrics": [{"name": "PPL"}],
        }
        YAMLConfigParser.validate_config(doc)
        assert "backbone" in doc["recipe"]
        assert doc["recipe"]["backbone"][0]["name"] == "Calibration"

    def test_recipe_both_name_and_backbone_raises(self):
        # Component form ({backbone: ...}) with a stray top-level "name" is
        # rejected by the schema's extra="forbid".
        with pytest.raises(RuntimeError, match="Extra inputs are not permitted"):
            YAMLConfigParser.validate_config(
                {
                    "model": {
                        "model_id": "x",
                        "sequence_length": 32,
                        "context_length": 64,
                    },
                    "recipe": {"name": "Calibration", "backbone": {"name": "Skip"}},
                    "metrics": [{"name": "PPL"}],
                }
            )

    def test_spinquant_no_rotations_enabled_raises(self):
        with pytest.raises(RuntimeError, match="SpinQuant"):
            YAMLConfigParser.validate_config(
                {
                    "model": {
                        "model_id": "x",
                        "sequence_length": 32,
                        "context_length": 64,
                    },
                    "recipe": [
                        {"name": "SpinQuant", "enable_r1": False, "enable_r2": False},
                        {"name": "Calibration"},
                    ],
                    "metrics": [{"name": "PPL"}],
                }
            )

    def test_spinquant_bare_raises(self):
        # A bare `{"name": "SpinQuant"}` no longer implies R1: it must be a
        # validation error rather than an implicit rotation.
        with pytest.raises(RuntimeError, match="SpinQuant"):
            YAMLConfigParser.validate_config(
                {
                    "model": {
                        "model_id": "x",
                        "sequence_length": 32,
                        "context_length": 64,
                    },
                    "recipe": [
                        {"name": "SpinQuant"},
                        {"name": "Calibration"},
                    ],
                    "metrics": [{"name": "PPL"}],
                }
            )

    def test_spinquant_backbone_only_on_vlm_raises(self):
        # SpinQuant on backbone but missing from visual is rejected: the schema
        # requires the pre-sim prefix to be identical across all components.
        with pytest.raises(RuntimeError, match="Pre-sim prefix of component"):
            YAMLConfigParser.validate_config(
                {
                    "model": {
                        "model_id": "x",
                        "sequence_length": 32,
                        "context_length": 64,
                    },
                    "recipe": {
                        "backbone": [
                            {"name": "SpinQuant", "enable_r1": True},
                            {"name": "Calibration"},
                        ],
                        "visual": [{"name": "Calibration"}],
                    },
                    "metrics": [{"name": "PPL"}],
                }
            )

    def test_spinquant_not_first_visual_step_raises(self):
        # SpinQuant must be the first visual step on a VLM: a pre-sim step after
        # an on-sim step breaks the required pre-sim-first ordering.
        with pytest.raises(RuntimeError, match="appears after an on-sim step"):
            YAMLConfigParser.validate_config(
                {
                    "model": {
                        "model_id": "x",
                        "sequence_length": 32,
                        "context_length": 64,
                    },
                    "recipe": {
                        "backbone": [
                            {"name": "SpinQuant", "enable_r1": True},
                            {"name": "Calibration"},
                        ],
                        "visual": [
                            {"name": "Calibration"},
                            {"name": "SpinQuant", "enable_r1": True},
                        ],
                    },
                    "metrics": [{"name": "PPL"}],
                }
            )


# ---------------------------------------------------------------------------
# Full parse_document (requires mocking detect_model_type)
# ---------------------------------------------------------------------------


class TestParseDocument:
    @pytest.fixture(autouse=True)
    def _setup_registry(self):
        """Register minimal recipes and metrics for parse tests."""
        from GenAILab.qai_hub_lm.schema.recipe import (
            RemoveQuantizationSpec,
            CalibrationSpec,
        )
        from GenAILab.qai_hub_lm.schema.dataset import C4Spec

        @YAMLConfigParser.register_recipe(RemoveQuantizationSpec)
        class RemoveQuantization:
            @staticmethod
            def apply(quantsim, generator, dataloader, **kwargs):
                pass

        @YAMLConfigParser.register_recipe(CalibrationSpec)
        class Calibration:
            @staticmethod
            def apply(quantsim, generator, dataloader, num_iterations=20, **kwargs):
                pass

        @YAMLConfigParser.register_metric
        class PPL:
            pass

        @YAMLConfigParser.register_metric
        class TinyMMLU:
            pass

        @YAMLConfigParser.register_dataset(C4Spec)
        class C4:
            pass

        class FakeLLM:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_minimal(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "metrics": [{"name": "TinyMMLU"}],
        }
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))
        assert result.model.model_type == "llama"
        assert result.metrics[0].metric_cls.__name__ == "TinyMMLU"

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_with_recipe_and_dataset(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "recipe": {
                "backbone": {
                    "name": "Calibration",
                    "dataset": {"name": "C4", "split": "en"},
                }
            },
            "metrics": [{"name": "PPL"}],
        }
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))
        assert result.recipe.backbone[0].technique_cls.__name__ == "Calibration"
        assert result.recipe.backbone[0].dataset_cls.__name__ == "C4"

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_unknown_recipe_raises(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "recipe": {"backbone": {"name": "NonexistentRecipe"}},
            "metrics": [{"name": "PPL"}],
        }
        with pytest.raises((LookupError, RuntimeError), match="NonexistentRecipe"):
            YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_unknown_metric_raises(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "metrics": [{"name": "FakeMetric"}],
        }
        with pytest.raises(LookupError, match="FakeMetric"):
            YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_export_true(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "metrics": [{"name": "PPL"}],
            "export": True,
        }
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))
        assert isinstance(result.export, str)

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_export_false(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "metrics": [{"name": "PPL"}],
            "export": False,
        }
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))
        assert not result.export

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_precision_parsed(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "metrics": [{"name": "PPL"}],
            "precision": {"activations": 8, "kv_cache": 4},
        }
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))
        from GenAILab.bench.precision import int8, int4

        assert result.precision.activations == int8
        assert result.precision.kv_cache == int4

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_unrecognized_section_raises(self, mock_detect, tmp_path):
        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "metrics": [{"name": "PPL"}],
            "unknown_section": {"foo": "bar"},
        }
        with pytest.raises(ValueError, match="Unrecognized"):
            YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="test_model")
    def test_export_enforces_required_adaptations(self, mock_detect, tmp_path):
        """Exporting without a required adaptation raises ValueError."""

        @YAMLConfigParser.register_adaptation(
            "RequiredAdapt", model_type="test_model", required_for_export=True
        )
        class RequiredMixin:
            pass

        # Register the model so use_dynamo_export can be checked
        @YAMLConfigParser.register_model("test_model")
        class TestModel:
            @staticmethod
            def use_dynamo_export():
                return True

        # The default LLM name must contain "ONNX" for the export path to trigger
        class FakeLLM_ONNX:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM_ONNX

        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "metrics": [{"name": "PPL"}],
        }
        with pytest.raises(ValueError, match="RequiredAdapt"):
            YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="test_model")
    def test_export_required_adaptation_skipped_with_exclusive(
        self, mock_detect, tmp_path
    ):
        """An exclusive adaptation suppresses required-for-export enforcement."""

        @YAMLConfigParser.register_adaptation(
            "RequiredAdapt", model_type="test_model", required_for_export=True
        )
        class RequiredMixin:
            pass

        @YAMLConfigParser.register_adaptation(
            "FullPipeline", model_type="test_model", exclusive=True
        )
        class FullPipelineMixin:
            pass

        @YAMLConfigParser.register_model("test_model")
        class TestModel:
            @staticmethod
            def use_dynamo_export():
                return True

        class FakeLLM_ONNX:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM_ONNX

        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
                "adaptations": ["FullPipeline"],
            },
            "metrics": [{"name": "PPL"}],
        }
        # Should NOT raise — exclusive adaptation bypasses the check
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))

    @pytest.mark.parametrize("fake_llm_name", ["FakeLLM_ONNX", "FakeLLM_Torch"])
    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_spinquant_extracted_from_chain(self, mock_detect, fake_llm_name, tmp_path):
        """SpinQuant is pulled out of the chain into 'spinquant' for both frameworks."""

        YAMLConfigParser._default_llm_cls = type(fake_llm_name, (), {})

        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "recipe": [
                {"name": "SpinQuant", "enable_r1": True, "enable_r2": False},
                {"name": "Calibration"},
            ],
            "metrics": [{"name": "PPL"}],
        }
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))

        # SpinQuant flags are extracted, step is stripped from the chain.
        assert result.recipe.pre_sim[0].recipe_kwargs == {
            "enable_r1": True,
            "enable_r2": False,
        }
        assert result.recipe.backbone[0].technique_cls.__name__ == "Calibration"

    @patch.object(YAMLConfigParser, "detect_model_type", return_value="llama")
    def test_no_spinquant_yields_none(self, mock_detect, tmp_path):
        """Without a SpinQuant step, 'spinquant' is None."""

        class FakeLLM_ONNX:
            pass

        YAMLConfigParser._default_llm_cls = FakeLLM_ONNX

        doc = {
            "model": {
                "model_id": "org/model",
                "sequence_length": 32,
                "context_length": 64,
            },
            "recipe": [{"name": "Calibration"}],
            "metrics": [{"name": "PPL"}],
        }
        result = YAMLConfigParser.parse_document(doc, export_base_dir=str(tmp_path))
        assert len(result.recipe.pre_sim) == 0


# ---------------------------------------------------------------------------
# ModelConfig.report_modifiers — the snapshot recorded under the report's
# ``model_modifiers`` field. Derived from the parsed config so it stays
# independent of the kwargs used to instantiate the model.
# ---------------------------------------------------------------------------


class TestReportModifiers:
    @staticmethod
    def _model_config(**overrides):
        base = dict(
            model_cls=object,
            model_id="org/model",
            model_type="llama",
            context_length=64,
            sequence_length=32,
            adaptations=[],
            extra_kwargs={},
        )
        base.update(overrides)
        return ModelConfig(**base)

    def test_records_adaptations(self):
        """adaptations are baked into the model class, not an instantiation
        kwarg, so they must be added explicitly (this was the regression)."""
        cfg = self._model_config(adaptations=["SHA", {"AIHM": {"foo": 1}}])
        assert cfg.report_modifiers()["adaptations"] == [
            "SHA",
            {"AIHM": {"foo": 1}},
        ]

    def test_includes_extra_kwargs_and_shape(self):
        cfg = self._model_config(
            context_length=4096,
            sequence_length=[1, 128],
            extra_kwargs={"num_hidden_layers": 2},
        )
        mods = cfg.report_modifiers()
        assert mods["num_hidden_layers"] == 2
        assert mods["context_length"] == 4096
        assert mods["sequence_length"] == [1, 128]

    def test_optional_fields_omitted_when_unset(self):
        mods = self._model_config().report_modifiers()
        assert "image_size" not in mods
        assert "encodings" not in mods
        assert "dtype" not in mods

    def test_optional_fields_included_when_set(self):
        cfg = self._model_config(
            image_size=[224, 224], encodings="/enc", dtype="float16"
        )
        mods = cfg.report_modifiers()
        assert mods["image_size"] == [224, 224]
        assert mods["encodings"] == "/enc"
        assert mods["dtype"] == "float16"

    def test_dtype_override_wins(self):
        """ONNX resolves an unset dtype to float32 and passes it explicitly."""
        cfg = self._model_config(dtype=None)
        assert cfg.report_modifiers(dtype="float32")["dtype"] == "float32"

    def test_does_not_mutate_extra_kwargs(self):
        extra = {"num_hidden_layers": 2}
        cfg = self._model_config(extra_kwargs=extra)
        cfg.report_modifiers()
        assert extra == {"num_hidden_layers": 2}
