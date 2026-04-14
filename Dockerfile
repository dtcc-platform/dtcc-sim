FROM condaforge/mambaforge:latest

ARG DTCC_TETGEN_WRAPPER_REF=main
ARG TETGEN_VERSION=v1.6.0

RUN mamba install -y -c conda-forge \
    fenics-dolfinx \
    petsc4py \
    compilers \
    cmake \
    ninja \
    git \
    curl \
    rsync && \
    mamba clean -afy

WORKDIR /app
COPY . /app

RUN git clone --depth 1 --branch "${DTCC_TETGEN_WRAPPER_REF}" \
    https://github.com/dtcc-platform/dtcc-tetgen-wrapper.git \
    /tmp/dtcc-tetgen-wrapper && \
    TETGEN_VERSION="${TETGEN_VERSION}" bash /tmp/dtcc-tetgen-wrapper/vendor_tetgen.sh && \
    perl -0pi -e 's/return py::array_t<double>\(\{0, m\}\);/return py::array_t<double>(py::array::ShapeContainer{0, m});/g; s/return py::array_t<int>\(\{0, m\}\);/return py::array_t<int>(py::array::ShapeContainer{0, m});/g; s/py::array_t<double> A\(\{n, m\}\);/py::array_t<double> A(py::array::ShapeContainer{n, m});/g; s/py::array_t<int> A\(\{n, m\}\);/py::array_t<int> A(py::array::ShapeContainer{n, m});/g; s/py::array_t<int> A\(\{n\}\);/py::array_t<int> A(py::array::ShapeContainer{n});/g; s/py::array_t<double> A\(\{n\}\);/py::array_t<double> A(py::array::ShapeContainer{n});/g' /tmp/dtcc-tetgen-wrapper/dtcc_tetgen_wrapper/cpp/tetwrap/tetwrap.cpp && \
    pip install --no-cache-dir /tmp/dtcc-tetgen-wrapper && \
    rm -rf /tmp/dtcc-tetgen-wrapper

RUN pip install --no-cache-dir -e ".[service]"

EXPOSE 8001

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]
