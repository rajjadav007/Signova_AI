from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error explanation")
    details: Optional[Any] = Field(default=None, description="Optional contextual or validation details")


class ErrorResponse(BaseModel):
    success: bool = Field(default=False, description="Success flag, always false for error responses")
    error: ErrorDetail
