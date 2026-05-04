"""Tests para construcción de modelos."""
import pytest


class TestModelBuilders:

    def test_resnet50_output_shape(self):
        """ResNet50 debe tener output shape (None, 1) con sigmoid."""
        from src.models.resnet50 import build_resnet50
        model = build_resnet50(input_shape=(224, 224, 3))
        assert model.output_shape == (None, 1)

    def test_densenet121_output_shape(self):
        from src.models.densenet121 import build_densenet121
        model = build_densenet121(input_shape=(224, 224, 3))
        assert model.output_shape == (None, 1)

    def test_vgg16_output_shape(self):
        from src.models.vgg16 import build_vgg16
        model = build_vgg16(input_shape=(224, 224, 3))
        assert model.output_shape == (None, 1)

    def test_backbone_named_correctly(self):
        """El backbone debe tener el nombre correcto para freeze/unfreeze."""
        from src.models.resnet50 import build_resnet50, BACKBONE_NAME
        model = build_resnet50()
        layer_names = [l.name for l in model.layers]
        assert BACKBONE_NAME in layer_names

    def test_head_architecture(self):
        """La cabeza debe tener GAP → Dense(128) → Dropout → Dense(1)."""
        from src.models.resnet50 import build_resnet50
        model = build_resnet50()
        layer_names = [l.name for l in model.layers]
        assert "gap" in layer_names
        assert "dense_head" in layer_names
        assert "dropout_head" in layer_names
        assert "output" in layer_names

    def test_freeze_backbone(self):
        """Después de freeze_backbone, solo la cabeza debe ser entrenable."""
        from src.models.resnet50 import build_resnet50, BACKBONE_NAME
        from src.models.base_model import freeze_backbone
        model = build_resnet50()
        freeze_backbone(model, BACKBONE_NAME)
        backbone = model.get_layer(BACKBONE_NAME)
        assert not backbone.trainable

    def test_unfreeze_full_backbone(self):
        """Después de unfreeze_backbone(from_layer=0), todo debe ser entrenable."""
        from src.models.resnet50 import build_resnet50, BACKBONE_NAME
        from src.models.base_model import freeze_backbone, unfreeze_backbone
        model = build_resnet50()
        freeze_backbone(model, BACKBONE_NAME)
        unfreeze_backbone(model, BACKBONE_NAME, from_layer=0)
        backbone = model.get_layer(BACKBONE_NAME)
        assert backbone.trainable

    def test_compile_model(self):
        """El modelo compilado debe tener las métricas correctas."""
        from src.models.resnet50 import build_resnet50
        from src.models.base_model import compile_model, freeze_backbone
        model = build_resnet50()
        freeze_backbone(model, "resnet50")
        compile_model(model, learning_rate=1e-3)
        metric_names = [m.name for m in model.metrics]
        assert "precision" in metric_names
        assert "recall" in metric_names
