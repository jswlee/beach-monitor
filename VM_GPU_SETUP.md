# GPU-Enabled Deployment on Ubuntu 24.04 VMs

This guide explains how to run the **Beach Monitor** API + agent stack on a **CUDA-enabled Ubuntu 24.04 VM**.

The goal: a VM where the API container uses the GPU (CUDA) for model inference, and the Streamlit/agent container stays CPU-only.

---

## 1. High-Level Overview

On the VM **host**, you only install:

- NVIDIA GPU **driver** (so the OS sees the GPU)
- **Docker** (to run containers)
- **NVIDIA Container Toolkit** (so Docker containers can access the GPU)

All app dependencies (PyTorch, Ultralytics, OpenCV, etc.) are installed **inside containers** via:

- `api/Dockerfile.cuda` (GPU-enabled API image)
- `agent/Dockerfile` (Streamlit/agent image)
- `docker-compose.cuda.yml` (ties them together with GPU reservations for the API only)

Once the host is set up, running the app is essentially:

```bash
git clone <your-repo-url> beach-monitor
cd beach-monitor
docker compose -f docker-compose.cuda.yml up --build -d
```

---

## 2. Prerequisites on the VM

You need a VM that:

- Has an **NVIDIA GPU attached** (via passthrough or a GPU flavor from your cloud provider)
- Is running **Ubuntu 24.04** (or compatible)
- Has internet access to pull Docker images and Python wheels

Log into the VM via SSH and perform all steps below there.

---

## 3. Install NVIDIA Driver on Ubuntu 24.04

1. **Update the system:**

```bash
sudo apt update && sudo apt upgrade -y
```

2. **Install the recommended NVIDIA driver:**

```bash
sudo ubuntu-drivers autoinstall
```

This installs a suitable `nvidia-driver-XXX` package for your GPU.

3. **Reboot the VM:**

```bash
sudo reboot
```

4. **Verify the GPU is visible after reboot:**

```bash
nvidia-smi
```

You should see a table listing the GPU, driver version, and CUDA version. If `nvidia-smi` fails or shows no GPU, fix that **before** proceeding.

---

## 4. Install Docker Engine

You do **not** need Docker Desktop on a Linux VM; just the Docker Engine.

1. **Install Docker via the convenience script:**

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

2. **Add your user to the `docker` group** so you can run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

3. **Test Docker:**

```bash
docker run --rm hello-world
```

If this prints the Hello World message, Docker is working.

---

## 5. Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker to pass the GPU into containers.

1. **Add the NVIDIA repository key and list:**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2. **Install the toolkit:**

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

3. **Configure Docker to use the NVIDIA runtime:**

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. **Verify GPU access from inside Docker:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

You should see output similar to `nvidia-smi` on the host, showing the GPU. If this fails, fix it before trying to run Beach Monitor.

---

## 6. Clone the Repository on the VM

From your home directory or a workspace folder on the VM:

```bash
git clone <your-repo-url> beach-monitor
cd beach-monitor
```

Replace `<your-repo-url>` with the actual Git URL of this repo.

---

## 7. Configure Environment Variables

The Docker Compose file `docker-compose.cuda.yml` expects several environment variables for AWS, segmentation models, and LangChain/OpenAI:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `S3_MODEL_KEY`
- `SEG_S3_BUCKET_NAME`
- `SEG_S3_CONFIG_KEY`
- `SEG_S3_WEIGHTS_KEY`
- `OPENAI_API_KEY`
- `LANGCHAIN_API_KEY` (optional)
- `LANGCHAIN_TRACING_V2` (optional)
- `LANGCHAIN_PROJECT` (optional)

You can provide these either via:

1. **Shell environment before running Compose:**

```bash
export AWS_ACCESS_KEY_ID=... \
       AWS_SECRET_ACCESS_KEY=... \
       S3_BUCKET_NAME=... \
       S3_MODEL_KEY=... \
       SEG_S3_BUCKET_NAME=... \
       SEG_S3_CONFIG_KEY=... \
       SEG_S3_WEIGHTS_KEY=... \
       OPENAI_API_KEY=...

# Optional LangChain vars
export LANGCHAIN_API_KEY=...
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="beach-monitor"
```

2. **A `.env` file** in the repo root (if you want to maintain a file-based config). Docker Compose will automatically read it if present. Example:

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=...
S3_MODEL_KEY=...
SEG_S3_BUCKET_NAME=...
SEG_S3_CONFIG_KEY=...
SEG_S3_WEIGHTS_KEY=...
OPENAI_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=beach-monitor
```

Make sure this file is **not committed** if it contains secrets.

---

## 8. Start the GPU-Enabled Stack

With Docker, the NVIDIA toolkit, and env vars in place, start the stack using the CUDA compose file:

```bash
cd beach-monitor

docker compose -f docker-compose.cuda.yml up --build -d
```

This will:

- Build `api/Dockerfile.cuda` into a GPU-enabled API image
- Build `agent/Dockerfile` into the Streamlit/agent image
- Start both services on a shared Docker network (`beach-monitor-network`)

Key settings from `docker-compose.cuda.yml`:

- `api-service` uses `api/Dockerfile.cuda`
- `deploy.resources.reservations.devices` configures the NVIDIA GPU for the API container
- `agent-service` is CPU-only, talking to the API via `http://api-service:8000`

---

## 9. Accessing the Services

From your local machine (or any client that can reach the VM’s IP):

- **API docs (FastAPI):**
  - `http://<VM-IP>:8000/docs`

- **Streamlit / Agent UI:**
  - `http://<VM-IP>:8501`

If you’re behind a cloud provider, make sure the relevant firewall/security group rules allow inbound traffic on ports **8000** and **8501**.

---

## 10. Verifying CUDA Inside the API Container

To confirm that the API container is actually using the GPU, exec into the running container and query PyTorch:

```bash
docker exec -it beach-monitor-api-cuda \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

A healthy GPU setup should output something like:

```text
True 1 NVIDIA GeForce RTX 50XX
```

If `torch.cuda.is_available()` is `False`, double-check:

- `docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi` on the host
- That `api/Dockerfile.cuda` is being used (via `docker-compose.cuda.yml`)
- That you didnt override `--gpus` or the `deploy.resources` section

---

## 11. Stopping and Restarting the Stack

- **Stop all containers:**

```bash
cd beach-monitor
docker compose -f docker-compose.cuda.yml down
```

- **Restart after code/config changes:**

```bash
cd beach-monitor
docker compose -f docker-compose.cuda.yml up --build -d
```

The `--build` flag ensures any changes to Dockerfiles or requirements are applied.

---

## 12. Summary

Once the VM has:

1. A working NVIDIA driver (`nvidia-smi` works)
2. Docker Engine installed
3. NVIDIA Container Toolkit configured (`docker run --rm --gpus all ... nvidia-smi` works)

Then deploying **Beach Monitor** with GPU acceleration is just:

```bash
cd beach-monitor
# ensure env vars / .env are set
docker compose -f docker-compose.cuda.yml up --build -d
```

The API will use CUDA for model inference, and the Streamlit agent UI will remain a separate, CPU-only service that calls into the GPU-enabled API.
