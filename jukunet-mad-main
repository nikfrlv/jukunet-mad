# ===============================================================
# Importing necessary Julia packages
# ===============================================================

using Random
using MatrixDepot
using DifferentialEquations
using LinearAlgebra
using Statistics

# ===============================================================
# Declaring network parameters
# ===============================================================

rng = MersenneTwister(58959); # random generator object

const N = 1000       # network size
const omega = 0      # central natural frequency for each layer
const d = 2          # width of a uniform frequency distributions for each layer
const omega_1 = [omega + d*rand(rng) - d/2 for x in range(1, stop=N)] # identical distributions for each layer

const A = matrixdepot("erdrey", Int8, N, 6*N) # Erdos-Renyi adjacency matrix with <k> = 12
const degree = sum(A, dims=1)[1, :];           # nodes degree

# ===============================================================
# Parameters of simulation times
# ===============================================================

dt = 0.01 # time step
dts = 0.1 # save time
ti = 0.0 
tt = 100.0
tf = 1000.0
nt = Int(div(tt,dts));
nf = Int(div(tf,dts));
tspan = (ti, tf); # time interval

# ===============================================================
# Monolayer Kuramoto model
# ===============================================================

function kuramoto_mono_frac_comp(du, u, p, t)
    u1 = @view u[1:N]
    du1 = @view du[1:N]
    
    z1 = Array{Complex{Float64},1}(undef, N)
    
    mul!(z1, A, exp.((u1)im))
    z1 = z1 ./ degree
    
    D = Array{Float64,1}(undef, N)
    @. D[1:floor(Int,p[1]*N)] = (1 - abs(z1[1:floor(Int,p[1]*N)]))
    @. D[1+floor(Int,p[1]*N):N] = abs(z1[1+floor(Int,p[1]*N):N])
    
    @. du1  = omega_1 + p[2] * D * degree * imag(z1 * exp((-1im) * u1))
    end;

# ===============================================================
# Bilayer Kuramoto model
# ===============================================================

function kuramoto_bi_frac_comp(du, u, p, t)
    u1 = @view u[1:N]
    u2 = @view u[N+1:2*N]
    
    du1 = @view du[1:N]
    du2 = @view du[N+1:2*N]
    
    z1 = Array{Complex{Float64},1}(undef, N)
    z2 = Array{Complex{Float64},1}(undef, N)
    
    mul!(z1, A, exp.((u1)im))
    mul!(z2, A, exp.((u2)im))
    
    z1 = z1 ./ degree
    z2 = z2 ./ degree
    
    D1 = Array{Float64,1}(undef, N)
    D2 = Array{Float64,1}(undef, N)
    @. D1[1:floor(Int,p[1]*N)] = (1 - abs(z2[1:floor(Int,p[1]*N)]))
    @. D1[1+floor(Int,p[1]*N):N] = abs(z2[1+floor(Int,p[1]*N):N])
    @. D2[1:floor(Int,p[1]*N)] = (1 - abs(z1[1:floor(Int,p[1]*N)]))
    @. D2[1+floor(Int,p[1]*N):N] = abs(z1[1+floor(Int,p[1]*N):N])
    
    @. du1  = omega_1 + p[2] * D1 * degree * imag(z1 * exp((-1im) * u1))
    @. du2  = omega_1 + p[2] * D2 * degree * imag(z2 * exp((-1im) * u2))
    end;
    
# ===============================================================
# Calculating Averaged global order parameter
# ===============================================================

function averaged_global_order(_u)
    global niter,novp,N

    _R_global = zeros(nf-nt+1)

    for n in nt:nf
        _re = mean(j -> cos(_u[j,n]), 1:N)
        _im = mean(j -> sin(_u[j,n]), 1:N)
        _R_global[n-nt+1] = sqrt(_re^2 + _im^2)
    end

    return mean(_R_global)
    end;
    
# ===============================================================
# Monolayer problem performance test
# ===============================================================

rng = MersenneTwister(58959); # random generator object
u0 = [2*pi*rand(rng) for x in range(1, stop=N)]
lambda = 0.3
f = 0.0

prob = ODEProblem(kuramoto_mono_frac_comp, u0, tspan, [f,lambda])
@time sol = solve(prob, RK4(), dt=dt, saveat=dts); # Fixed-step 4th order Runge-Kutta method

# Output: 1.331904 seconds (158.41 k allocations: 1.442 GiB, 8.03% gc time)
# MacBook-Pro 2019, 2,6 GHz 6-Core Intel Core i7

# ===============================================================
# Bilayer problem performance test
# ===============================================================

L = 2
rng = MersenneTwister(58959); # random generator object
u0 = [2*pi*rand(rng) for x in range(1, stop=L*N)]
lambda = 0.3
f = 0.0

prob = ODEProblem(kuramoto_bi_frac_comp, u0, tspan, [f,lambda])
@time sol = solve(prob, RK4(), dt=dt, saveat=dts); # Fixed-step 4th order Runge-Kutta method

# Output: 2.887773 seconds (305.68 k allocations: 2.872 GiB, 12.74% gc time)
# MacBook-Pro 2019, 2,6 GHz 6-Core Intel Core i7

# ===============================================================
# ES boundaries (Monolayer problem)
# ===============================================================

rng = MersenneTwister(58959); # random generator object

# arranging coupling strength
lambda1 = LinRange(0.10,0.25,61)
f = 0.0

# arranging global order parameter arrays
R_av_global_f_comp1 = ones(length(lambda1))
R_av_global_b_comp1 = ones(length(lambda1))

# Forward Transition
u0 = [2*pi*rand(rng) for x in range(1, stop=2*N)]
for l1 in 1:length(lambda1)
    println(lambda1[l1])
    prob = ODEProblem(kuramoto_mono_frac_comp, u0, tspan, [f,lambda1[l1]])
    sol = solve(prob, RK4(), dt=dt, saveat=dts);
    R_av_global_f_comp1[l1] = averaged_global_order(sol[1:N,:])
    u0 = sol[:,Int(tf*dts)+1]
    end;

# Backward Transition
u0 = [2*pi*rand(rng) for x in range(1, stop=2*N)]
for l1 in length(lambda1):-1:1
    println(lambda1[l1])
    prob = ODEProblem(kuramoto_mono_frac_comp, u0, tspan, [f,lambda1[l1]])
    sol = solve(prob, RK4(), dt=dt, saveat=dts);
    R_av_global_b_comp1[l1] = averaged_global_order(sol[1:N,:])
    u0 = sol[:,Int(tf*dts)+1]
    end;
    
# ===============================================================
# ES boundaries (Bilayer problem)
# ===============================================================

rng = MersenneTwister(58959); # random generator object

# arranging coupling strength
lambda1 = LinRange(0.10,0.25,61)
f = 0.0
L = 2

# arranging global order parameter arrays
R_av_global_f_comp1 = ones(length(lambda1))
R_av_global_b_comp1 = ones(length(lambda1))
R_av_global_f_comp2 = ones(length(lambda1))
R_av_global_b_comp2 = ones(length(lambda1))

# Forward transition
u0 = [2*pi*rand(rng) for x in range(1, stop=L*N)]
for l1 in 1:length(lambda1)
    prob = ODEProblem(kuramoto_bi_frac_comp, u0, tspan, [f,lambda1[l1]])
    sol = solve(prob, RK4(), dt=dt, saveat=dts);
    R_av_global_f_comp1[l1] = averaged_global_order(sol[1:N,:])
    R_av_global_f_comp2[l1] = averaged_global_order(sol[N+1:L*N,:])
    u0 = sol[:,Int(tf*dts)+1]
    end;

# Backward transition
u0 = [2*pi*rand(rng) for x in range(1, stop=L*N)]
for l1 in length(lambda1):-1:1
    prob = ODEProblem(kuramoto_bi_frac_comp, u0, tspan, [f,lambda1[l1]])
    sol = solve(prob, RK4(), dt=dt, saveat=dts);
    R_av_global_b_comp1[l1] = averaged_global_order(sol[1:N,:])
    R_av_global_b_comp2[l1] = averaged_global_order(sol[N+1:L*N,:])
    u0 = sol[:,Int(tf*dts)+1]
    end;
