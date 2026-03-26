#!/bin/bash
set -euo pipefail

# A handy script to run the AMIP benchmark on CSD3 interactively.
# Good to verify if the setup is working before more expensive sbatch NSight
# runs.


CLIMA_COUPLER=ClimaCoupler.jl
RUN_NAME=baseline
OUTPUT_DIR=results/PCR-AMIP

PROJECT_DIR=$CLIMA_COUPLER/experiments/ClimaEarth

AMIP_CONFIG=$CLIMA_COUPLER/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
ATMOS_CONFIG=$CLIMA_COUPLER/config/atmos_configs/climaatmos_progedmf_1m.yml
CONFIG_FILE=$AMIP_CONFIG

# Load modules
module purge
module load rhel8/cclake/base
module load julia/1.11.4
module load cuda/12.1


# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib


# Instantiate julia environment, precompile, and build CUDA
julia --project=$PROJECT_DIR -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); Pkg.status()'

# Downgrade the CUDA version used by julia to match the version on CSD3
julia --project=$PROJECT_DIR -e 'using CUDA; CUDA.set_runtime_version!(v"12.1")'

julia --project=$PROJECT_DIR \
    scripts/run.jl \
    --config=$CONFIG_FILE
