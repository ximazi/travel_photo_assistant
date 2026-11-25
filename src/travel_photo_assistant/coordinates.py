"""Coordinate system and preview viewport transformation chain.

Implements README.txt Section VI and integration_notes.txt Section IV:
- Canonical analysis coordinate space: upright, unmirrored normalized [0, 1],
  origin at top-left, x right, y down.
- Maps normalized coordinates through aspect-fit (contain) or aspect-fill (cover)
  scaling/cropping and optional horizontal preview mirroring into viewport pixels.
- Enforces transform_version matching to prevent displaying stale overlay boxes.
- Keeps input_image, photographer_body, and subject_body direction references separate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreviewTransformConfig:
    """Defines how the upright unmirrored input image is displayed in the preview view."""

    input_width_px: int
    input_height_px: int
    viewport_width_px: float
    viewport_height_px: float
    viewport_fit: str = "contain"  # "contain" (aspect-fit) | "cover" (aspect-fill)
    preview_mirrored: bool = False
    transform_version: int = 1


@dataclass(frozen=True)
class ViewportRect:
    """Pixel rectangle in preview screen coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    transform_version: int
    clipped: bool = False

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


class TransformVersionMismatchError(ValueError):
    """Raised when an overlay coordinate belongs to an outdated transform version."""


class PreviewMapper:
    """Maps canonical upright-unmirrored [0, 1] coordinates to preview pixels."""

    def __init__(self, config: PreviewTransformConfig) -> None:
        if config.input_width_px <= 0 or config.input_height_px <= 0:
            raise ValueError("Input dimensions must be positive.")
        if config.viewport_width_px <= 0 or config.viewport_height_px <= 0:
            raise ValueError("Viewport dimensions must be positive.")
        if config.viewport_fit not in {"contain", "cover"}:
            raise ValueError(f"Unsupported viewport_fit: {config.viewport_fit}")
        self.config = config

        scale_x = config.viewport_width_px / config.input_width_px
        scale_y = config.viewport_height_px / config.input_height_px
        self._scale = min(scale_x, scale_y) if config.viewport_fit == "contain" else max(scale_x, scale_y)
        self._drawn_w = config.input_width_px * self._scale
        self._drawn_h = config.input_height_px * self._scale
        self._offset_x = (config.viewport_width_px - self._drawn_w) / 2.0
        self._offset_y = (config.viewport_height_px - self._drawn_h) / 2.0

    def _check_version(self, transform_version: int | None) -> None:
        if transform_version is not None and transform_version != self.config.transform_version:
            raise TransformVersionMismatchError(
                f"Stale transform_version={transform_version}, active={self.config.transform_version}"
            )

    def map_point(
        self,
        point: list[float] | tuple[float, float],
        *,
        transform_version: int | None = None,
    ) -> tuple[float, float]:
        """Map a normalized [x, y] point from upright unmirrored input to viewport pixels."""
        self._check_version(transform_version)
        x_norm, y_norm = float(point[0]), float(point[1])
        if self.config.preview_mirrored:
            x_norm = 1.0 - x_norm
        px = self._offset_x + x_norm * self._drawn_w
        py = self._offset_y + y_norm * self._drawn_h
        return (round(px, 2), round(py, 2))

    def map_bbox(
        self,
        bbox: list[float] | tuple[float, float, float, float],
        *,
        transform_version: int | None = None,
        clamp_to_viewport: bool = True,
    ) -> ViewportRect:
        """Map a normalized [xmin, ymin, xmax, ymax] bbox into preview viewport pixels."""
        self._check_version(transform_version)
        xmin, ymin, xmax, ymax = (float(v) for v in bbox)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(f"Invalid bbox ordering: {bbox}")

        p1 = self.map_point((xmin, ymin))
        p2 = self.map_point((xmax, ymax))
        vx_min = min(p1[0], p2[0])
        vx_max = max(p1[0], p2[0])
        vy_min = min(p1[1], p2[1])
        vy_max = max(p1[1], p2[1])

        clipped = False
        if clamp_to_viewport:
            clamped_x_min = max(0.0, min(self.config.viewport_width_px, vx_min))
            clamped_x_max = max(0.0, min(self.config.viewport_width_px, vx_max))
            clamped_y_min = max(0.0, min(self.config.viewport_height_px, vy_min))
            clamped_y_max = max(0.0, min(self.config.viewport_height_px, vy_max))
            clipped = (
                clamped_x_min != vx_min
                or clamped_x_max != vx_max
                or clamped_y_min != vy_min
                or clamped_y_max != vy_max
            )
            vx_min, vx_max, vy_min, vy_max = clamped_x_min, clamped_x_max, clamped_y_min, clamped_y_max

        return ViewportRect(
            x_min=round(vx_min, 2),
            y_min=round(vy_min, 2),
            x_max=round(vx_max, 2),
            y_max=round(vy_max, 2),
            transform_version=self.config.transform_version,
            clipped=clipped,
        )

    def map_horizontal_direction(self, direction: str, direction_reference: str) -> str:
        """Map left/right UI arrows & text consistently under preview mirroring.

        - `input_image`: flips left <-> right when `preview_mirrored=True` so arrows
          and screen-relative instructions stay consistent with what the user sees.
        - `photographer_body` and `subject_body`: anatomical/body directions NEVER flip
          due to display mirroring; they must stay explicitly tied to body frame.
        """
        norm_dir = direction.lower().strip()
        if norm_dir not in {"left", "right", "up", "down", "none"}:
            return norm_dir

        if direction_reference == "input_image" and self.config.preview_mirrored:
            if norm_dir == "left":
                return "right"
            if norm_dir == "right":
                return "left"
        return norm_dir
