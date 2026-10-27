"""Domain exceptions. The API layer maps them to HTTP status codes; nothing else needs to know HTTP."""


class HelmetDetectionError(Exception):
    """Base class for all errors raised by the pipeline."""


class ModelLoadError(HelmetDetectionError):
    """A model file is missing or cannot be loaded."""


class InvalidMediaError(HelmetDetectionError):
    """The uploaded file is not a readable image/video, or violates size/length limits."""


class ProcessingError(HelmetDetectionError):
    """Unexpected failure while processing a valid input."""
