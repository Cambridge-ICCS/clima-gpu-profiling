using Revise
using Infiltrator
using Debugger

using ClimaCore
using CUDA
using ClimaComms

import ClimaCore:
    Utilities,
    Geometry,
    Domains,
    Meshes,
    Topologies,
    Hypsography,
    Spaces,
    Fields,
    Operators,
    Quadratures
import ClimaCore.MatrixFields: @name

using LazyBroadcast

import ClimaCore.MatrixFields: @name
import LinearAlgebra: ldiv!

redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(true)))

@show ClimaComms.device()
CUDA.allowscalar(true)

# Note: Taken from "/test/MatrixFields/matrix_field_test_utils.jl"
#
# Generate extruded finite difference spaces for testing. Include topography
# when possible.
function test_spaces(::Type{FT}; velem = 63, helem = 16, npoly = 3) where {FT}

    comms_ctx = ClimaComms.SingletonCommsContext(ClimaComms.device())
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

    # Always flat since we test with CUDA
    hypsography =  Hypsography.Flat() 
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace, hypsography)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

FT = Float32


# Instanciate spaces
center_space, face_space = test_spaces(FT)

# Build diffusion paramter fields 
ρ = FT(1.5)
k = FT(0.3)
cₚ = FT(100.0)

ᶜρ = @. $ones(FT, center_space) * ρ
ᶜk = @. $ones(FT, center_space) * k
ᶠk = @. $ones(FT, face_space) * k
ᶜcₚ = @. $ones(FT, center_space) * cₚ

b = @. $ones(FT, center_space) * FT(0.7) # Avoid special value (e.g. 1)

# Diffusion problem set-up
grad = ClimaCore.Operators.GradientC2F(;
    top = ClimaCore.Operators.SetValue(FT(0)),
    bottom = ClimaCore.Operators.SetValue(FT(0)),
)
grad_matrix_lazy = ClimaCore.MatrixFields.operator_matrix(grad)

div = ClimaCore.Operators.DivergenceF2C()
div_matrix_lazy = ClimaCore.MatrixFields.operator_matrix(div)

field_2_matrix = ClimaCore.MatrixFields.DiagonalMatrixRow

# Build a matrix
A = @. lazy_broadcast(
    -inv(field_2_matrix(ᶜρ * ᶜcₚ)) *
    div_matrix_lazy() *
    ClimaCore.MatrixFields.DiagonalMatrixRow(ᶠk) *
    grad_matrix_lazy(),
)

# This is a Hell of the problem...
# Type inference gaves up...
@show eltype(A) # eltype(A) = Any 


# At the moment we cannot build MatrixField on a lazy expression
#  that evaluates to a BandMatrixRow Field (aka field of matrix rows)
A_fieldmatrix =
    ClimaCore.MatrixFields.FieldMatrix((@name(T), @name(T)) => Base.materialize(A))
b_fieldvector = ClimaCore.Fields.FieldVector(; T = b)


A′ = ClimaCore.MatrixFields.FieldMatrixWithSolver(A_fieldmatrix, b_fieldvector)
x_fieldvector = similar(b_fieldvector)

ldiv!(x_fieldvector, A′, b_fieldvector)
