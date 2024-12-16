FROM langchain/langgraph-api:3.11

# Install git and other necessary tools including script utility
RUN apt-get update && apt-get install -y git bsdutils

RUN mkdir -p /repos

ADD . /deps/LM-Systems

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"engineer": "/deps/LM-Systems/src/langgraph_engineer/agent.py:graph"}'

# Set terminal environment variables
ENV TERM=xterm-256color
ENV COLUMNS=80
ENV LINES=24

WORKDIR /repos

RUN chmod 777 /repos