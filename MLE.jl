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

t = df.t


β0 = theta0[1:3]
β1 = theta0[4:6]
σ = theta0[11:36]
δz = theta0[7:8]
δt = theta0[9]
σ = exp.(σ)
ρ = theta0[10]

function mle(theta, df)
    β0 = theta[1:3]
    β1 = theta[4:6]
    σ = theta[11:36]
    δz = theta[7:8]
    δt = theta[9]
    σ = exp.(σ)
    ρ = theta[10]
    df0 = df[df[:school] .== 0, :]
    df1 = df[df[:school] .== 1, :]
    dfsel = by(df, :caseid, x -> mean(x.school))
    Y0 = df0.y
    Y1 = df1.y
    X0 = [df0.xa df0.xb df0.xb.*df0.xb]
    X1 = [df1.xa df1.xb df1.xb.*df1.xb]
    Z = [unique(df,:caseid).za unique(df,:caseid).zb]
    epsilon0s = Y0 .- X0*β0
    epsilon1s = Y1 .- X1*β1
    Xs = [df.xa df.xb df.xb.*df.xb df.xbc df.xbc.*df.xbc][ df.t .== 1, :]
    for i = 2:T
        Xt = [df.xa df.xb df.xb.*df.xb df.xbc df.xbc.*df.xbc][ df.t .== i, :]
        global Xs = Xt .+ Xs
    end
    sel = Xs[:,1:3]*β1 .- [Xs[:,1] Xs[:,4:5]]*β0 .- Z*δz
    contrib0 = 0.
    contrib1 = 0.
    nodes, weights = gausshermite(100)
    for t = 1:T
        for i = 1:4352
            integ1 = sum((pdf.(Normal(), (epsilon1s[ df1.t .== t][i] .- ρ*σ[2*T+2].*nodes)./σ[T+t])./σ[T+t]).*(cdf.(Normal(), (sel[dfsel.x1 .== 1][i] .- σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights)
            contrib1 = contrib1 + log(integ1)
        end
        for j = 1:5648
            integ0 = sum((pdf.(Normal(), (epsilon0s[ df0.t .== t][j] .- ρ*σ[2*T+2].*nodes)./σ[t])./σ[t]).*(1 .- cdf.(Normal(), (sel[dfsel.x1 .== 0][j] .- σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights)
            contrib0 = contrib0 + log(integ0)
        end
    end
    return ll = -contrib0 - contrib1
end
integ1 = sum((pdf.(Normal(), (epsilon1s[ df1.t .== 2][222] .- ρ*σ[2*T+2].*nodes)./σ[T+2])./σ[T+2]).*(cdf.(Normal(), (sel[dfsel.x1 .== 1][222] .- σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights)


β0 = [1. 2. -0.01]
β1 = [.9 3.4 -0.01]
σ = transpose(repeat([1.1],2*T+2))
δz = [5.1 3.1]
δt = 0.5
ρ = 0.8

theta = [β0 β1 δz δt ρ σ]

mle(theta,df)



@time mini = optimize(vars -> mle(vars, df), theta0, BFGS())
