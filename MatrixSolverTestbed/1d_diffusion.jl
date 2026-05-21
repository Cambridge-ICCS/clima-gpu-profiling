using Revise
using Infiltrator
using Debugger

using ClimaCore
using CUDA
using ClimaComms


using LazyBroadcast

import ClimaCore.MatrixFields: @name
import LinearAlgebra: ldiv!

redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(true)))

@show ClimaComms.device()
CUDA.allowscalar(true)


FT = Float32

domain = ClimaCore.Domains.IntervalDomain(
    ClimaCore.Geometry.ZPoint(FT(-1.0)),
    ClimaCore.Geometry.ZPoint(FT(1.0)),
    (:bottom, :top),
)

mesh = ClimaCore.Meshes.IntervalMesh(domain; nelems = 100)
topology = ClimaCore.Topologies.IntervalTopology(mesh)
center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(topology)
face_space = ClimaCore.Spaces.FaceFiniteDifferenceSpace(topology)


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
