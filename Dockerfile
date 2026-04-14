FROM condaforge/mambaforge:latest

RUN mamba install -y -c conda-forge fenics-dolfinx petsc4py compilers && mamba clean -afy

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir -e /workspace && \
    pip install --no-cache-dir -e "/workspace/temp/dtcc-sim[service]"

WORKDIR /workspace/temp/dtcc-sim

EXPOSE 8001

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]
