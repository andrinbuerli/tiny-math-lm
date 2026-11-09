FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Node.js and development tools
RUN apt-get update && \
    apt-get install -y \
      curl git ca-certificates build-essential \
      && curl -fsSL https://deb.nodesource.com/setup_22.x \
      | bash - \
      && apt-get install -y nodejs \
      && npm install -g @openai/codex \
      && rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} codexgroup && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} \
    -d /home/codex -s /bin/bash codex

WORKDIR /workspace

CMD ["bash"]

