"""Benchmark tests for FieldMatrix tridiagonal solves.

"""

using Pkg
using CUDA
using ClimaComms
using ClimaCore
using BenchmarkTools

import ClimaCore.Fields
import ClimaCore.DataLayouts
using LinearAlgebra: Tridiagonal, norm
using Statistics: mean


redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(true)))

# Figure out which project is currently activated
project_dir = dirname(Base.active_project())
@info "Active project: $project_dir"


const clima_core_path = begin
    uuid = get(
        () -> error("`ClimaCore` not found in the project dependencies."),
        Pkg.project().dependencies,
        "ClimaCore",
    )
    abspath(Pkg.dependencies()[uuid].source)
end


include(joinpath(clima_core_path, "test", "MatrixFields", "matrix_field_test_utils.jl"))

# Generate extruded finite difference spaces for testing. Include topography
# when possible.
function test_spaces(::Type{FT}; velem = 63, helem = 16, npoly = 3) where {FT}

    comms_ctx = ClimaComms.SingletonCommsContext(comms_device)
    hdomain = Domains.SphereDomain(FT(10))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(comms_ctx, hmesh)
    quad = Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(10));
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vtopology = Topologies.IntervalTopology(comms_ctx, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    sfc_coord = Fields.coordinate_field(hspace)
    hypsography =
        using_cuda ? Hypsography.Flat() :
        Hypsography.LinearAdaption(
            Geometry.ZPoint.(@. cosd(sfc_coord.lat) + cosd(sfc_coord.long) + 1),
        ) # TODO: FD operators don't currently work with hypsography on GPUs.
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace, hypsography)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

# Create a field matrix for a similar solve to ClimaAtmos's moist dycore + prognostic,
# EDMF + prognostic surface temperature with implicit acoustic waves and SGS fluxes
# also returns corresponding FieldVector
function dycore_prognostic_EDMF_FieldMatrix(
    ::Type{FT},
    center_space = nothing,
    face_space = nothing,
) where {FT}
    seed!(1) # For reproducibility with random fields
    if isnothing(center_space) || isnothing(face_space)
        center_space, face_space = test_spaces(FT)
    end
    surface_space = Spaces.level(face_space, half)
    sfc_vec = random_field(FT, surface_space)
    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    λ = 10
    ᶜᶜmat1 = random_field(DiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space) ./ λ
    ᶠᶜmat2 = random_field(BidiagonalMatrixRow{FT}, face_space) ./ λ
    ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)
    # Geometry.Covariant123Vector(1, 2, 3) * Geometry.Covariant12Vector(1, 2)'
    e¹² = Geometry.Covariant12Vector(1, 1)
    e₁₂ = Geometry.Contravariant12Vector(1, 1)
    e³ = Geometry.Covariant3Vector(1)
    e₃ = Geometry.Contravariant3Vector(1)

    ρχ_unit = (; ρq_tot = 1, ρq_liq = 1, ρq_ice = 1, ρq_rai = 1, ρq_sno = 1)
    ρaχ_unit = (; ρaq_tot = 1, ρaq_liq = 1, ρaq_ice = 1, ρaq_rai = 1, ρaq_sno = 1)

    ᶠᶜmat2_u₃_scalar = ᶠᶜmat2 .* (e³,)
    ᶜᶠmat2_scalar_u₃ = ᶜᶠmat2 .* (e₃',)
    ᶠᶠmat3_u₃_u₃ = ᶠᶠmat3 .* (e³ * e₃',)
    ᶜᶠmat2_ρχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit ⊠ e₃')), ᶜᶠmat2)
    ᶜᶜmat3_uₕ_scalar = ᶜᶜmat3 .* (e¹²,)
    ᶜᶜmat3_uₕ_uₕ =
        ᶜᶜmat3 .* (
            Geometry.Covariant12Vector(1, 0) * Geometry.Contravariant12Vector(1, 0)' +
            Geometry.Covariant12Vector(0, 1) * Geometry.Contravariant12Vector(0, 1)',
        )
    ᶜᶠmat2_uₕ_u₃ = ᶜᶠmat2 .* (e¹² * e₃',)
    ᶜᶜmat3_ρχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit)), ᶜᶜmat3)
    ᶜᶜmat3_ρaχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit)), ᶜᶜmat3)
    ᶜᶠmat2_ρaχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit ⊠ e₃')), ᶜᶠmat2)

    dry_center_gs_unit = (; ρ = 1, ρe_tot = 1, uₕ = e¹²)
    center_gs_unit = (; dry_center_gs_unit..., ρatke = 1, ρχ = ρχ_unit)
    center_sgsʲ_unit = (; ρa = 1, ρae_tot = 1, ρaχ = ρaχ_unit)

    b = Fields.FieldVector(;
        sfc = sfc_vec .* ((; T = 1),),
        c = ᶜvec .* ((; center_gs_unit..., sgsʲs = (center_sgsʲ_unit,)),),
        f = ᶠvec .* ((; u₃ = e³, sgsʲs = ((; u₃ = e³),)),),
    )
    A = MatrixFields.FieldMatrix(
        # GS-GS blocks:
        (@name(c.ρe_tot), @name(c.ρe_tot)) => ᶜᶜmat3,
        (@name(c.ρatke), @name(c.ρatke)) => ᶜᶜmat3,
        (@name(c.ρχ), @name(c.ρχ)) => ᶜᶜmat3,
        (@name(c.uₕ), @name(c.uₕ)) => ᶜᶜmat3_uₕ_uₕ,
        (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
        # GS-SGS blocks:
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρae_tot)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_tot), @name(c.sgsʲs.:(1).ρaχ.ρaq_tot)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_liq), @name(c.sgsʲs.:(1).ρaχ.ρaq_liq)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_ice), @name(c.sgsʲs.:(1).ρaχ.ρaq_ice)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_rai), @name(c.sgsʲs.:(1).ρaχ.ρaq_rai)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_sno), @name(c.sgsʲs.:(1).ρaχ.ρaq_sno)) => ᶜᶜmat3,
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
        (@name(c.ρatke), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
        (@name(c.ρχ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_ρχ_scalar,
        (@name(c.uₕ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_uₕ_scalar,
        (@name(f.u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
        # SGS-SGS blocks:
        (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
    )
    return A, b
end


"""
    ulp_distance(a,b)

Calculates the absolute error between `a` and `b` in the units of ulp(b)
If b is power of two then the ulp is computed in the direction of a.

Return Inf for not finite numbers.

We return a Float since the ulp distance may not be an integer when
they a and b are in diffrent binades.
"""
function ulp_distance(a::FT, b::FT) where {FT<:Base.IEEEFloat}
    if !isfinite(a) || !isfinite(b)
        return Inf
    end
    # ulp_b is always exact by Sterbenz Lemma
    ulp_b = a < b ? b - prevfloat(b) : nextfloat(b) - b
    return abs(a - b) / ulp_b
end


function test_field_matrix_solver(; test_name, alg, A, b, use_rel_error = false)
    # @testset "$test_name" begin
    x = similar(b)
    A′ = FieldMatrixWithSolver(A, b, alg)
    @test zero(A′) isa typeof(A′)
    solve_time = @benchmark ClimaComms.@cuda_sync comms_device ldiv!(x, A′, b)

    b_test = similar(b)
    # @test zero(b) isa typeof(b)
    mul_time = @benchmark ClimaComms.@cuda_sync comms_device mul!(b_test, A′, x)

    solve_time_rounded = round(solve_time; sigdigits = 2)
    mul_time_rounded = round(mul_time; sigdigits = 2)
    time_ratio = solve_time_rounded / mul_time_rounded
    time_ratio_rounded = round(time_ratio; sigdigits = 2)

    error_vector = abs.(parent(b_test) .- parent(b))
    if use_rel_error
        rel_error = norm(error_vector) / norm(parent(b))
        rel_error_rounded = round(rel_error; sigdigits = 2)
        error_string = "Relative Error = $rel_error_rounded"
    else
        max_error = maximum(error_vector)
        max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))
        error_string = "Maximum Error = $max_eps_error eps"
    end

    @info "$test_name:\n\tSolve Time = $solve_time_rounded s, \
           Multiplication Time = $mul_time_rounded s (Ratio = \
           $time_ratio_rounded)\n\t$error_string"

    if use_rel_error
        @test rel_error < 1e-5
    else
        @test max_eps_error <= 3
    end

    # TODO: fix broken test when Nv is added to the type space
    using_cuda || @test @allocated(ldiv!(x, A′, b)) ≤ 1536
    using_cuda || @test @allocated(mul!(b_test, A′, x)) == 0
    # end
end



function reference_solve(A, b; reference_precision = BigFloat)
    # Unpack matrix field into matrix
    Ni, Nj, _, _, Nh = DataLayouts.universal_size(Fields.field_values(A))
    n_batch = Ni * Nj * Nh

    # I presume a number of vertical levels
    # TODO: I think we should be able to read it from the Universal size
    Nv = DataLayouts.nlevels(Fields.field_values(A))

    # Construct host-based tridiagonal matrix
    # Why off diagonals still have size 63?
    Am1, A0, A1 =
        ClimaCore.MatrixFields.unzip_tuple_field_values(Fields.field_values(A.entries))


    Am1 = reshape(parent(Am1), Nv, n_batch)
    A0 = reshape(parent(A0), Nv, n_batch)
    A1 = reshape(parent(A1), Nv, n_batch)
    b_flat = reshape(parent(Fields.field_values(b)), Nv, n_batch)

    # Host based result Matrix
    x = Matrix(similar(b_flat))

    FT = reference_precision

    for j = 1:n_batch
        v_Am1 = Vector{FT}(Am1[2:end, j])
        v_A0 = Vector{FT}(A0[:, j])
        v_A1 = Vector{FT}(A1[1:(end-1), j])
        v_b = Vector{FT}(b_flat[:, j])

        A_tridiag = Tridiagonal(v_Am1, v_A0, v_A1)

        # Converst back to the element precision
        x[:, j] = A_tridiag \ v_b
    end

    return x
end


"""
Benchmark a tridiagonal solver function on the GPU and compare to a reference solution.

Return the solutions from the function so they can be easiliy inspected in case
the errors a re high.
"""
function benchmark_tridiagonal_solver(
    solver_function,
    A,
    b;
    case_name,
    cache = nothing,
    reference_precision = BigFloat,
)
    x = similar(b)
    @info "Name: $case_name"
    @info "BenchmarkTools run"
    # There is another `@benchmark` macro from the test utils... we need to be explicit about module
    benchmark_result =
        BenchmarkTools.@benchmark CUDA.@sync $solver_function($cache, $x, $A, $b)
    display(benchmark_result)


    @info "CUDA-timed/profiled run"
    # Warmup 'compilation run`, should not trigger anything after `@benchmark` run
    CUDA.@time solver_function(cache, x, A, b)

    # Clean the result space
    x = similar(b)
    gpu_time = CUDA.@elapsed CUDA.@profile external = true solver_function(cache, x, A, b)

    # Flatten the solution along the horizontal columns to match
    # the reference solution shape
    x_flat_host = begin
        Ni, Nj, _, _, Nh = DataLayouts.universal_size(Fields.field_values(A))
        n_batch = Ni * Nj * Nh
        Nv = DataLayouts.nlevels(Fields.field_values(A))
        Matrix(reshape(parent(Fields.field_values(x)), Nv, n_batch))
    end

    # Verify accuracy of the result
    x_ref = reference_solve(A, b; reference_precision)

    ulp_error = ulp_distance.(x_flat_host, x_ref)
    max_ulp_error = maximum(ulp_error)
    mean_ulp_error = mean(ulp_error)

    l2_error = norm(x_flat_host - x_ref)

    @info "Name: $case_name, gpu_time: $gpu_time [s], size: $(size(parent(A)))"
    @info "Max ULP error: $max_ulp_error, Mean ULP error: $mean_ulp_error"
    @info "L2 error: $l2_error"
    return x_flat_host, x_ref
end


#############################################################
# Benchmarking of tridiagonal matrix solver


# @testset "FieldMatrixSolver Unit Tests" begin
FT = Float32

velem = 63 # Vertical elements
helem = 16 # Horizontal elements
npoly = 3 # Polynomial order

center_space, face_space = test_spaces(FT; velem, helem, npoly)
surface_space = Spaces.level(face_space, half)

# Note that the sequence is not type stable!
# If you change FT type you get basically independent samples
seed!(1) # ensures reproducibility

ᶜvec = random_field(FT, center_space)
ᶠvec = random_field(FT, face_space)
sfc_vec = random_field(FT, surface_space)

# Make each random square matrix diagonally dominant in order to avoid large
# large roundoff errors when computing its inverse. Scale the non-square
# matrices by the same amount as the square matrices.
λ = 10 # scale factor
ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)

# Realistic 'block matrix' case for full solver
#A, b = dycore_prognostic_EDMF_FieldMatrix(FT, center_space, face_space)

# We need to pick functions from the extension module
# Make it avaliable
ClimaCoreCUDAExt = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)

x_sol, x_ref = benchmark_tridiagonal_solver(
    (cache, x, A, b) ->
        ClimaCoreCUDAExt.single_field_solve!(ClimaComms.device(), cache, x, A, b),
    ᶜᶜmat3,
    ᶜvec;
    case_name = "Baseline (local mem Thomas alg)",
    # Cache is not used... but is touched (unpacked)
    # We need to provide it
    cache = ClimaCore.MatrixFields.single_field_solver_cache(ᶜᶜmat3, ᶜvec),
    reference_precision = FT,
)
