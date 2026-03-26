#!/bin/bash
#SBATCH -J dycore-nsys
#SBATCH -A ICCS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --qos=INTR


# NOTE: This script must be run from the root of the repository!
#       It relies on some hardcoded relative paths.
#       e.g. submit with `sbatch scripts/run-ncu-csd3.sh`



set -euo pipefail

CLIMA_COUPLER=ClimaCoupler.jl
RUN_NAME=baseline
OUTPUT_DIR=results/PCR-AMIP

PROJECT_DIR=$CLIMA_COUPLER/experiments/ClimaEarth

AMIP_CONFIG=$CLIMA_COUPLER/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
ATMOS_CONFIG=$CLIMA_COUPLER/config/atmos_configs/climaatmos_progedmf_1m.yml
CONFIG_FILE=$AMIP_CONFIG

# Ensure the output prefix parent directory exists
mkdir -p "$OUTPUT_DIR"

# Load modules
module purge
module load rhel8/cclake/base
module load julia/1.11.4
module load cuda/12.1

# Fix the bug in nsys with julia
LD_LIBRARY_PATH=$(julia --startup-file=no -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))'):$LD_LIBRARY_PATH

# Set environment variables for GPU usage
export CLIMACOMMS_DEVICE=CUDA
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true

# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Make sure we will use the local packages
julia scripts/instantiate_clima_earth.jl

# Instantiate julia environment, precompile, and build CUDA
julia --project=$PROJECT_DIR -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); Pkg.status()'

# Downgrade the CUDA version used by julia to match the version on CSD3
julia --project=$PROJECT_DIR -e 'using CUDA; CUDA.set_runtime_version!(v"12.1")'

nsys profile \
    --capture-range=cudaProfilerApi \
    --kill=none \
    --trace=nvtx,cuda,osrt \
    --gpu-metrics-device=all \
    --cuda-memory-usage=true \
    --output=$OUTPUT_DIR/$RUN_NAME \
    julia --project=$PROJECT_DIR \
    scripts/run.jl \
    --config=$CONFIG_FILE
