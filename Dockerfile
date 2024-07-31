FROM codercom/code-server:latest

WORKDIR /home/coder/project

# Copy files outside of "src/"
COPY requirements3.txt /home/coder/project/
COPY .git/ /home/coder/project/.git
COPY README.md /home/coder/project/
COPY .gitignore /home/coder/project/

# Install system dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    libxml2-dev \
    libxslt-dev \
    && sudo rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install dependencies
RUN sudo python3 -m venv /home/coder/project/venv
RUN sudo /home/coder/project/venv/bin/pip install jupyterlab
RUN sudo /home/coder/project/venv/bin/pip install -r requirements3.txt
RUN sudo /home/coder/project/venv/bin/pip install ipykernel
RUN sudo /home/coder/project/venv/bin/python -m ipykernel install --user --name=project_venv

# Install VS Code extensions
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension ms-toolsai.jupyter

# Change ownership of the directory
RUN sudo chown -R coder:coder /home/coder/project

# Set Git configurations
RUN git config --global user.email "vbafna@umich.edu" && \
    git config --global user.name "Vaibhav"


CMD ["/home/coder/start.sh", "--auth", "none", "/home/coder/project"]