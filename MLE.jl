using Pkg
using Random
using Distributions
using LinearAlgebra
using Optim
using Statistics
using DataFrames
using CSVFiles
using Tables
using FastGaussQuadrature

df = load("fakedata.csv") |> DataFrame
global T = maximum(df.age[df[:caseid] .== 1, :]) - minimum(df.age[df[:caseid] .== 1, :]) + 1

t = repeat(1:12,10000)
df = [df DataFrame(t=t) DataFrame(xbc = df.xb .- 4)]
df.xbc[df.school .== 1] = df.xbc[df.school .== 1] .+ 8


S = df.school

t = df.t

β0 = [1, 2]
β1 = [1.1, 2.1]
σ = repeat([1.1],2*T+1)
δz = [1.1, 1.1]


function mle(params, df)
    β0 = params[1:2]
    β1 = params[3:4]
    σ = params[7:31]
    δz = params[5:6]
    σ = exp.(σ)
    df0 = df[df[:school] .== 0, :]
    df1 = df[df[:school] .== 1, :]
    Y0 = df0.y
    Y1 = df1.y
    X0 = [df0.xa df0.xb]
    X1 = [df1.xa df1.xb]
    Z = [unique(df,:caseid).za unique(df,:caseid).zb]
    epsilon0s = Y0 .- X0*β0
    epsilon1s = Y1 .- X1*β1
    Xs = [df.xa df.xb df.xbc][ df.t .== 1, :]
    for i = 2:T
        Xt = [df.xa df.xb df.xbc][ df.t .== i, :]
        Xs = Xt .+ Xs
    end
    sel = Xs[:,1:2]*β1 .- [Xs[:,1] Xs[:,3]]*β0 .- Z*δz
    selection0s = log.(pdf.(Normal(), epsilon0s[ df0.t .== 1]./σ[1])) .- log(σ[1])
    selection1s =  log.(pdf.(Normal(), epsilon1s[ df1.t .== 1]./σ[T+1])) .- log(σ[T+1])
    for i = 2:T
        selection0t = log.(pdf.(Normal(), epsilon0s[ df0.t .== i]./σ[i])) .- log(σ[i])
        selection0s = selection0s .+ selection0t
        selection1t = log.(pdf.(Normal(), epsilon1s[ df1.t .== i]./σ[T+i])) .- log(σ[T+i])
        selection1s = selection1s .+ selection1t
    end
    ll = - sum(T.*(1 .- log.(cdf.(Normal(),(sel) ./ σ[2*T+1])))) - sum(selection0s) - sum( T.*log.(cdf.(Normal(), (sel) ./ σ[2*T+1]))) - sum(selection1s)
end
β0 = [1 2]
β1 = [1.1 2.1]
σ = transpose(repeat([1.1],2*T+1))
δz = [1.1 1.1]

theta0 = [β0 β1 δz σ]
mini = optimize(vars -> mle(vars, df), theta0, BFGS())


Optim.minimizer(mini)
Optim.minimizer(mini)[7:31]= exp.(Optim.minimizer(mini)[7:31]).^2
σ = params[7:31]
