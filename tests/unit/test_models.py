import tensorflow as tf

from multi_agent_ml_ops.models.model_registry import ModelRegistry
from multi_agent_ml_ops.models.text_classifier import TextClassifierModel
from multi_agent_ml_ops.models.sentiment_model import SentimentModel


def test_text_classifier_model_exports_saved_model(tmp_path):
    model = TextClassifierModel()
    save_dir = tmp_path / "classifier"
    model.save(save_dir)
    loaded = tf.saved_model.load(str(save_dir))
    assert loaded is not None


def test_sentiment_model_predicts_expected_shape():
    model = SentimentModel()
    result = model.predict(["great movie", "bad movie"])
    assert result.shape[0] == 2


def test_model_registry_loads_by_version_tag(tmp_path):
    model = TextClassifierModel()
    save_dir = tmp_path / "classifier"
    model.save(save_dir)
    registry = ModelRegistry(root_dir=tmp_path)
    loaded = registry.load_model("classifier", version="latest")
    assert loaded is not None
