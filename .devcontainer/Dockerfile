FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install development tools and utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    sqlite3 \
    curl \
    ca-certificates \
    sudo \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (NodeSource) for CLI/node development
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude and Codex CLIs globally
RUN npm install -g @anthropic-ai/claude-code @openai/codex

# Install common tooling globally
RUN npm install -g pnpm

# Build arguments for user/group configuration
ARG USER_UID=1000
ARG USER_GID=1000

# Create vscode group with matching GID
RUN groupadd -g ${USER_GID} vscode 2>/dev/null || true

# Create vscode user with matching UID
RUN useradd -u ${USER_UID} -g ${USER_GID} -m -s /bin/bash vscode

# Fix permissions on home directory
RUN chown -R ${USER_UID}:${USER_GID} /home/vscode

# Give vscode passwordless sudo
RUN echo "vscode ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Switch to non-root user
USER vscode

# Set working directory
WORKDIR /workspace
