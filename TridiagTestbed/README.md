# How to run

First make sure that the `ClimaCore.jl` submodule is checked out to the revision 
you wish to test.

The instantiate the project here with the standard:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```
the project is configured to pick up the local `ClimaCore.jl` sources in the 
root of the repository/.

Then you can run the tests with:
```bash
julia --project ./benchmark_tridiag_solve.jl
```

Don't forget to set `CLIMA_COMMS_DEVICE=CUDA` to run on a GPU. Otherwise 
the solution will be performed on the host.


The ULP comparison is not working well at the moment due to catastrophic 
cancellation in the solution. ULP counts can be hight if the reference value is 
low (in the order of ϵ). 

Happy developing!  
