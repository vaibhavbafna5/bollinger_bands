# Use the code-server image as base
FROM codercom/code-server:latest

# Set the working directory
WORKDIR /workspace

# Copy the contents of the current directory to the container
COPY . /workspace

# Install python
RUN sudo apt-get update && \
    sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    && sudo rm -rf /var/lib/apt/lists/*

# Install Jupyter Lab globally
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install jupyterlab && \
    pip install -r requirements3.txt && \
    deactivate

# Install tailscale
RUN sudo curl -fsSL https://tailscale.com/install.sh | sh

# Install VS Code extensions
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension ms-toolsai.jupyter

# Create config.yaml with password and bind-addr for code-server
RUN echo "bind-addr: 0.0.0.0:8080" > /home/coder/.config/code-server/config.yaml && \
    echo "password: open" >> /home/coder/.config/code-server/config.yaml

# Expose the ports needed for Tailscale and code-server
EXPOSE 8080
