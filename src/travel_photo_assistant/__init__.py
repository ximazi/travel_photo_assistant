"""Travel Photo Assistant v0.1 core package."""

from .config import AssetPaths, RuntimeConfig
from .controller import FastLoopResult, SlowLoopResult, TravelPhotoController
from .coordinates import PreviewMapper, PreviewTransformConfig, ViewportRect
from .knowledge import RetrievalMode, ShotCardRepository, ShotCardRetriever
from .schema_validator import ProviderSchemaAdapter, SchemaValidator
from .validators import BusinessRuleValidator, ValidationIssue, ValidationResult

__all__ = [
    "AssetPaths",
    "RuntimeConfig",
    "TravelPhotoController",
    "SlowLoopResult",
    "FastLoopResult",
    "PreviewMapper",
    "PreviewTransformConfig",
    "ViewportRect",
    "RetrievalMode",
    "ShotCardRepository",
    "ShotCardRetriever",
    "ProviderSchemaAdapter",
    "SchemaValidator",
    "BusinessRuleValidator",
    "ValidationIssue",
    "ValidationResult",
]

__version__ = "0.1.0"
