# Stage 1: Build frontend
FROM node:20-slim AS frontend-builder

WORKDIR /build
COPY dtcc-dataset-downloader/frontend/package*.json ./
RUN npm ci
COPY dtcc-dataset-downloader/frontend/ ./
RUN npm run build


# Stage 2: Main application
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment with FEniCSx
RUN conda config --remove channels defaults 2>/dev/null; \
    conda config --add channels conda-forge && \
    conda create -n dtcc --override-channels -c conda-forge python=3.12 -y && \
    conda install -n dtcc --override-channels -c conda-forge fenics-dolfinx mpich pyvista -y && \
    conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "dtcc", "/bin/bash", "-c"]

# Install GDAL/fiona via conda (avoids building from source)
RUN conda install -n dtcc --override-channels -c conda-forge fiona gdal -y && conda clean -afy

# Install Rust (needed for dtcc-pyspade-native on ARM)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install dtcc-core and dtcc from develop branch
RUN pip install --no-cache-dir \
    git+https://github.com/dtcc-platform/dtcc-core.git@develop \
    git+https://github.com/dtcc-platform/dtcc.git@develop

# Build and install dtcc-tetgen-wrapper
COPY dtcc-tetgen-wrapper/ /tmp/tetgen/
RUN cd /tmp/tetgen && bash vendor_tetgen.sh && pip install --no-cache-dir . && rm -rf /tmp/tetgen

# Install dtcc-sim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY dtcc_sim/ ./dtcc_sim/
RUN pip install --no-cache-dir .

# Install dtcc-dataset-downloader (dtcc-core already satisfied from above)
COPY dtcc-dataset-downloader/pyproject.toml dtcc-dataset-downloader/README.md ./dtcc-dataset-downloader/
COPY dtcc-dataset-downloader/server/ ./dtcc-dataset-downloader/server/
COPY dtcc-dataset-downloader/publisher/ ./dtcc-dataset-downloader/publisher/
RUN cd dtcc-dataset-downloader && pip install --no-cache-dir .

# Copy built frontend
COPY --from=frontend-builder /build/dist/ ./dtcc-dataset-downloader/server/static/

# Copy sandbox scripts
COPY sandbox/ ./sandbox/

EXPOSE 8000

WORKDIR /app/dtcc-dataset-downloader
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dtcc"]
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
