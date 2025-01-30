from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from .configuration import Configuration
from langchain_core.runnables import RunnableConfig
from typing import Optional


def _get_model(config: Optional[RunnableConfig], default: str, key: str):
    # Get configuration instance
    config_instance = Configuration.from_runnable_config(config)

    # Get model type from config, defaulting to 'anthropic'
    model_type = config['configurable'].get('model', 'anthropic') if config and 'configurable' in config else 'anthropic'

    if model_type == "openai":
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
            api_key=config_instance.openai_api_key
        )
    elif model_type == "anthropic":
        return ChatAnthropic(
            temperature=0,
            model_name="claude-3-haiku-20240307",
            api_key=config_instance.anthropic_api_key
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
