#!/bin/bash
#SBATCH -J "iccs-nsys-amip"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00

set -euo pipefail
CLIMA_COUPLER=ClimaCoupler.jl
RUN_NAME=baseline
OUTPUT_DIR=results/AMIP_RUNS

PROJECT_DIR=$CLIMA_COUPLER/experiments/ClimaEarth
CONFIG_FILE=$CLIMA_COUPLER/config/benchmark_configs/amip_progedmf_1m_land_he16.yml

# Ensure the output prefix parent directory exists
mkdir -p "$OUTPUT_DIR"

# Load modules
module purge
module load climacommon/2025_05_15
module load julia/1.11.4
module load nsight-systems/2025.3.1

echo "Environment:"
module list

echo "Configure environment:"
# Downgrade the CUDA version used by julia to match the version on clima server
julia --project=$PROJECT_DIR -e 'using CUDA; CUDA.set_runtime_version!(v"12.9")'
# Fix the bug in nsys with julia
LD_LIBRARY_PATH=$(julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))'):$LD_LIBRARY_PATH

# Set environment variables for GPU usage
export CLIMACOMMS_DEVICE=CUDA
export GPU_COMPILER_MANGLE_STACK_TRACE=1
export GPU_COMPILER_MANGLE_IGNORED_MODULES=":Base,:Core,:GPUCompiler,:CUDA,:NVTX,:ClimaCoreCUDAExt,:ClimaCore"

# Set environmental variable for julia to not use global packages for reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Instantiate julia environment, precompile, and build CUDA
julia --project=$PROJECT_DIR -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

echo "Running nsys profiling script..."
nsys profile \
    --capture-range=cudaProfilerApi \
    --kill=none \
    --trace=nvtx,mpi,cuda,osrt \
    --gpu-metrics-devices=cuda-visible \
    --cuda-memory-usage=true \
    --output=$OUTPUT_DIR/$RUN_NAME \
    julia --project=$PROJECT_DIR \
    scripts/run.jl \
    --config=$CONFIG_FILE
