"""Dynamic, language-defined decisions and their shared scoring model."""

# Keep importing the data schema independent of optional torch dependencies.
from .schema import Decision, Option

__all__ = ["Decision", "Option"]
