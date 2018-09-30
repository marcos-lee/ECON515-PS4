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
df = [df DataFrame(t=t) DataFrame(xbc = df.xb .- 4) ]
df.xbc[df.school .== 1] = df.xbc[df.school .== 1] .+ 8
df = [df DataFrame(xb2 = df.xb.*df.xb) DataFrame(xbc2 = df.xbc.*df.xbc)]

dfsel = by(df, :caseid, x -> mean(x.school))



function mle(theta, df)
    β0 = theta[1:3]
    β1 = theta[4:6]
    σ = theta[11:36]
    δz = theta[7:8]
    δt = theta[9]
    σ = exp.(σ)
    ρ = theta[10]
    epsilon0s = df.y .- [df.xa df.xb df.xb2]*β0
    epsilon1s = df.y .- [df.xa df.xb df.xb2]*β1
    Xs = by(df, :caseid) do x
        DataFrame(xsa = sum(x.xa), xsb = sum(x.xb), xsb2 = sum(x.xb2), xsbc = sum(x.xbc), xsbc2 = sum(x.xbc2))
    end
    df = join(df, Xs, on=:caseid, kind=:inner)
    df = [df DataFrame(sel = repeat([transpose([df[:xsa] df[:xsbc] df[:xsbc2]][1,:])*β1 - transpose([df[:xsa] df[:xsb] df[:xsb2]][1,:])*β0],120000))]
    #for i = 1:120000 #very slow, there has to be a better way
    #    if df[:school][i] == 1
    #        df.sel[i] = transpose([df[:xsa] df[:xsb] df[:xsb2]][i,:])*β1 - transpose([df[:xsa] df[:xsbc] df[:xsbc2]][i,:])*β0
    #    else
    #        df.sel[i] = transpose([df[:xsa] df[:xsbc] df[:xsbc2]][i,:])*β1 - transpose([df[:xsa] df[:xsb] df[:xsb2]][i,:])*β0
    #    end
    #end
    df.sel = df.sel .- [df.za df.zb]*δz
    contrib0 = repeat([0.],120000)
    contrib1 = repeat([0.],120000)
    nodes, weights = gausshermite(100)
    for i = 1:120000
        t = df.t[i]
        integ1 = sum((pdf.(Normal(), (epsilon1s[i] .- ρ*σ[2*T+2].*nodes)./σ[T+t])./σ[T+t]).*(cdf.(Normal(), (df.sel[i] .- σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights)
        contrib1[i] = log(integ1)
        integ0 = sum((pdf.(Normal(), (epsilon0s[i] .- ρ*σ[2*T+2].*nodes)./σ[T+t])./σ[T+t]).*(1 .- cdf.(Normal(), (df.sel[i] .- σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights)
        contrib0[i] = log(integ0)
    end
    return ll = -sum(contrib0[df.school .== 0]) - sum(contrib1[df.school .==1])
end



β0 = [1. 2. -0.01]
β1 = [.9 3.4 -0.01]
σ = transpose(repeat([1.1],2*T+2))
δz = [5.1 3.1]
δt = 0.5
ρ = 0.8

theta = [β0 β1 δz δt ρ σ]

@time mle(theta,df)



@time mini = optimize(vars -> mle(vars, df), theta0)
@time mini1 = optimize(vars -> mle(vars, df), theta0, BFGS())
