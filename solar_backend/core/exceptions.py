"""Custom application exceptions."""


class SolarPlannerError(Exception):
    """Base exception for solar planner domain errors."""


class InferenceError(SolarPlannerError):
    """Raised when model inference fails."""


class ValidationError(SolarPlannerError):
    """Raised when user input validation fails."""
