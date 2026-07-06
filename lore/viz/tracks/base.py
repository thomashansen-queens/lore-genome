"""
Base classes for visualization tracks.

A track owns *what* to draw (features, a sequence, a plotted line) and *how tall*
it is. It does not own the horizontal coordinate math: that is a `Scale`, handed
in at render time. This keeps every track ignorant of broken axes, panning, and
alignment with its neighbours in a stack.
"""
from abc import ABC, abstractmethod
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING

from .. import svg as v
from .. import text
from ..scale import Break, Scale, TrackBounds

if TYPE_CHECKING:
    from .stack import TrackTheme


class LabelPosition(StrEnum):
    LEFT = "left"
    TOP = "top"
    NONE = "none"


class BaseTrack(BaseModel, ABC):
    """
    Abstract base for all tracks. Inherits BaseModel for input coercion and
    validation. Subclasses implement `render_payload()`.

    Breaks are discontinuities per track, which are merged in the stack to unify
    the coordinate system across tracks.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str
    label_pos: LabelPosition = LabelPosition.LEFT
    metadata: dict = Field(default_factory=dict)

    breaks: list[Break] = Field(default_factory=list)
    invert: bool = False

    theme_kwargs: dict = Field(default_factory=dict)

    def model_post_init(self, __context):
        """Pydantic lifecycle hook: funnel arbitrary kwargs into theme_kwargs."""
        if self.__pydantic_extra__:
            self.theme_kwargs.update(self.__pydantic_extra__)
            self.__pydantic_extra__.clear()

    def build_scale(
        self,
        domain: TrackBounds,
        px_width: float,
        shared_breaks: list[Break] | tuple[Break, ...] = (),
    ) -> Scale:
        """
        Compose the coordinate scale for this track. Override to give a track its
        own domain (e.g. a per-replicon synteny row); by default it adopts the
        stack's shared domain and merges shared + per-track breaks.
        """
        return Scale(domain, px_width, breaks=[*shared_breaks, *self.breaks])

    @abstractmethod
    def get_extents(self) -> tuple[float, float]:
        """Returns the (min, max) data coordinates present in this track"""
        ...

    @abstractmethod
    def render_payload(self, scale: Scale, theme: "TrackTheme") -> v.SvgGroup:
        """Draw the track's data into a group, using `scale` for all x-positioning."""
        ...
