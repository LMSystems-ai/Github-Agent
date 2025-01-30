from dataclasses import dataclass, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
import os

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the LangGraph Engineer."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # First check config, then environment variables
        values: dict[str, Any] = {
            f.name: configurable.get(f.name, os.environ.get(f.name.upper()))
            for f in fields(cls)
            if f.init
        }

        return cls(**{k: v for k, v in values.items() if v is not None})
