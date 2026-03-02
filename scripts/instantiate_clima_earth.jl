#!/usr/bin/env julia
# Instantiate the ClimaEarth.jl environment, and make sure that the ClimaCore.jl and ClimaAtmos.jl
# track the correct paths.
#
# If the  Manifest already exists, it will change the packages to track the local versions.
# If it doesn't exist, it will try to create it using the defaults and *then* modify
# the paths. Note that the resolution from scratch may fail...
#
using Pkg
using UUIDs

# constants
const clima_earth_path =
    joinpath(@__DIR__, "..", "ClimaCoupler.jl", "experiments", "ClimaEarth")
const root_dir = joinpath(@__DIR__, "..")

# We separate package path from Package name since they may (and are different)
# One reason is the `.jl` suffix, but renaming may be expected as well;
function make_package_track_path(package_name, package_path)
    uuid = get(
        () -> error("Package '$package_name' is not found in the environment"),
        Pkg.project().dependencies,
        package_name,
    )
    abs_package_path = abspath(package_path)


    # May be brittle, is marked 'experimental'
    package_info = Pkg.dependencies()[uuid]

    if !package_info.is_tracking_path
        # Make it track the path
        @info "Package '$package_name' was not tracking a path. Making it track '$abs_package_path'."
        Pkg.develop(path = abs_package_path)
        return
    end

    package_path_in_env = Pkg.dependencies()[uuid].source
    if !isabspath(package_path_in_env)
        error("Script assumes that the Pkg will give us absolute path. It did not...")
    end

    if !samefile(package_path_in_env, abs_package_path)
        @info "Package '$package_name' was tracking a different path: '$package_path_in_env'. Making it track '$abs_package_path'."
        Pkg.develop(path = abs_package_path)
    end
end


# Check if the path exists (e.g. submodules were not initialized)
if !isdir(clima_earth_path)
    error(
        "ClimaEarth not found at: '$clima_earth_path'. Have you initialised submodules: `git submodule update --init --recursive`?",
    )
end

# Save time. We don't want to precompile things in this script
ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0

Pkg.activate(clima_earth_path)

# If there is Manifest present: downloads the dependencies
# If there isn't: first resolves dependencies and then downloads them
Pkg.instantiate()

# Update existing Manifest with correct paths
make_package_track_path("ClimaCore", joinpath(root_dir, "ClimaCore.jl"))
make_package_track_path("ClimaAtmos", joinpath(root_dir, "ClimaAtmos.jl"))
