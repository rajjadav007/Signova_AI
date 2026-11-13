from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Application health status")
    application: str = Field(description="Application title or name")
    version: str = Field(description="Application version")
    environment: str = Field(description="Configured runtime environment")
