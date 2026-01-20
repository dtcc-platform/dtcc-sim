# Installation

## Create Python environment

    python3.12 -m venv venv
    source venv/bin/activate

## Install DTCC Platform

    pip install dtcc

## Install DCC TetGen wrapper

    git clone git@github.com:dtcc-platform/dtcc-tetgen-wrapper.git
    cd dtcc-tetgen-wrapper
    bash vendor_tetgen.sh
    pip install .
