#= This version of the `run.jl` is intended to be used interactivly

 Just invoke it with `julia -i ./scripts/run_intr.jl`
 and it will: 
  1) move to the AMIP environment
  2) load the right config file 
  3) Build the case and run single timestep to compile the kernels

It requires that the usual suspests are in your base environment! 
  - Revise 
  - Infiltrator
  - Debugger
  - Cthulhu
=#
using Pkg
# Move to the AMIP environment
amip_env  = joinpath(@__DIR__, "..", "ClimaCoupler.jl", "experiments", "AMIP")
Pkg.activate(amip_env)

# Load the usual suspects for interactive debugging
using Revise
using Infiltrator
using Debugger
using Cthulhu

config_file_path = joinpath(@__DIR__, "..", "ClimaCoupler.jl", "config", "benchmark_configs", "amip_progedmf_1m_land_he16.yml")


# Run the benchmark
import CUDA

# Figure out which project is currently activated and include the setup
# script
project_dir = dirname(Base.active_project())
@info "Active project: $project_dir"
include(joinpath(project_dir, "code_loading.jl"))

# Set up and run the coupled simulation
cs = CoupledSimulation(config_file_path)

# Run a single step to compile
step!(cs)
