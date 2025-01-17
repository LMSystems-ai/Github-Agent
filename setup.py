from setuptools import setup, find_packages

setup(
    name="langgraph_engineer",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langgraph",
        "langchain_anthropic",
        "langchain_core",
        "langchain_openai",
        "gitpython",
        "python-dotenv",
        "pydantic",
        "aider-chat"
    ],
) 