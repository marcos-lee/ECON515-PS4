using Distributions
using LinearAlgebra
using Optim
using Statistics
using DataFrames
using CSVFiles
using FastGaussQuadrature
using NLSolversBase
using Plots
using KernelDensity

#df = load("fakedata.csv") |> DataFrame
#df = load("fake_data_julia.csv") |> DataFrame #i changed the variable name income to y
df = load("nlsy.csv") |> DataFrame #i changed the variable name income to y

#define global number of periods and unique individuals
const T = convert(Int64,maximum(df.age[df[:caseid] .== 1, :]) - minimum(df.age[df[:caseid] .== 1, :]) + 1)
const Ni = convert(Int64,maximum(df.caseid))

#create period variable
xb = zeros(Ni*T)
for i = 1:Ni*T
    if df.school[i] == 1
        xb[i] = df.age[i] - 18
    else
        xb[i] = df.age[i] - 22
    end
end
t = repeat(1:T,Ni)
df = [df DataFrame(t=t) DataFrame(xa = ones(Ni*T)) DataFrame(xb = xb)]
df = [df DataFrame(xbc = df.xb .- 4) ]

#create the counterfactual experience
df.xbc[df.school .== 1] = df.xbc[df.school .== 1] .+ 8
df = [df DataFrame(xb2 = df.xb.*df.xb) DataFrame(xbc2 = df.xbc.*df.xbc) ]

df = [df DataFrame(y = log.(df.earnings))]
#create choice specific dataframes, with its own unique individual caseid from 1 to Ni0
df0 = df[df.school .== 0,:]
sort!(df0, (:t, :caseid))
df0.caseid = repeat(1:size(df0[df0.t .== 1,:].caseid)[1],T)
sort!(df0, (:caseid, :t))

df1 = df[df.school .== 1,:]
sort!(df1, (:t, :caseid))
df1.caseid = repeat(1:size(df1[df1.t .== 1,:].caseid)[1],T)
sort!(df1, (:caseid, :t))


#selection equation sum of X variables across t=1:T
Xs = by(df, :caseid) do x
    DataFrame(xsa = sum(x.xa), xsb = sum(x.xb), xsb2 = sum(x.xb2), xsbc = sum(x.xbc), xsbc2 = sum(x.xbc2), za = mean(x.xa), zb = mean(x.tuit4), zc = mean(x.faminc79), zd = mean(x.numsibs), ze = mean(x.scores_cognitive_skill), school = mean(x.school))
end

#Number of unique individuals in each choice dataframe
const Ni0 = convert(Int64,maximum(df0.caseid))
const Ni1 = convert(Int64,maximum(df1.caseid))


include("MLE_NLSY_Functions.jl")


#I've coded up simple OLS and Probit estimators, to get sensible estimators
data0 = [df0.xa df0.xb df0.xb2]
data1 = [df1.xa df1.xb df1.xb2]

ols0 = OLS(df0.y, data0)
ols1 = OLS(df1.y, data1)

eps0 = log(sqrt(mean((df0.y .- data0 * ols0).^2)))
eps1 = log(sqrt(mean((df1.y .- data1 * ols1).^2)))

datap = [Xs.zb Xs.zc Xs.zd Xs.ze Xs.school]
thetap = [1. 1. 1. 1. 1.]
@time prest = optimize(vars -> probit(vars, datap), thetap, Optim.Options(iterations = 5000))



#let's put our initial guesses in an Array
β0 = [ols0[1], ols0[2], ols0[3]]
β1 = [ols1[1], ols1[2], ols1[3]]
σ0 = repeat([eps0],T)
σ1 = repeat([eps1],T)
σw = 0. #just use the guess from fake data
σt = log(sqrt(.4)) #just use the guess from fake data
δz = [Optim.minimizer(prest)[1], Optim.minimizer(prest)[2], Optim.minimizer(prest)[3], Optim.minimizer(prest)[4], Optim.minimizer(prest)[5]]
δt = 0.5 #just use the guess from fake data
ρ = 0.8 #just use the guess from fake data

theta = vcat(β0, β1, δz, δt, ρ, σ0, σ1, σw, σt)


#define nodes and weights of gauss hermite
nnodes = 20
nodes, weights = gausshermite(nnodes)
quad = [nnodes, nodes, weights]

#just check how much time it takes to evaluate
@time mle(theta,df0, df1, Xs, quad)


#this is the optimization routine.
@time mini = optimize(vars -> mle(vars, df0, df1, Xs, quad), theta, Optim.Options(iterations = 10000))


#unpack the estimates
β0 = Optim.minimizer(mini)[1:3]
β1 = Optim.minimizer(mini)[4:6]
σ = exp.(Optim.minimizer(mini)[14:23]).^2
δz = Optim.minimizer(mini)[7:11]
δt = Optim.minimizer(mini)[12]
ρ = Optim.minimizer(mini)[13]


θ = rand(Normal(0,σ[10]),Ni)
θp = repeat(θ, inner=T)
ϵ0 = zeros(Ni0*T)

ϵ1 = zeros(Ni1*T)
for i = 1:Ni0*T
    if df0.t[i] == 1
        ϵ0[i] = rand(Normal(0,σ[1]))
    elseif df0.t[i] == 2
        ϵ0[i] = rand(Normal(0,σ[2]))
    elseif df0.t[i] == 3
        ϵ0[i] = rand(Normal(0,σ[3]))
    else
        ϵ0[i] = rand(Normal(0,σ[4]))
    end
end

for i = 1:Ni1*T
    if df1.t[i] == 1
        ϵ1[i] = rand(Normal(0,σ[5]))
    elseif df0.t[i] == 2
        ϵ1[i] = rand(Normal(0,σ[6]))
    elseif df0.t[i] == 3
        ϵ1[i] = rand(Normal(0,σ[7]))
    else
        ϵ1[i] = rand(Normal(0,σ[8]))
    end
end

y0 = [df0.xa df0.xb df0.xb2]*β0 .+ ρ.*θp[df.school .== 0] .+ ϵ0
y1 = [df1.xa df1.xb df1.xb2]*β1 .+ θp[df.school .==1] .+ ϵ1

df0 = [df0 DataFrame(y0 = y0)]
df1 = [df1 DataFrame(y1 = y1)]
Ys0 = by(df0, :caseid) do x
    DataFrame(sumy0 = sum(x.y0), sumy = sum(x.y))
end
Ys1 = by(df1, :caseid) do x
    DataFrame(sumy1 = sum(x.y1), sumy = sum(x.y))
end

#plot the sum of log earnings for school = 0
y0_e = kde(y0)
x = range(7, stop = 14, length = 250) |> collect
plot(x, z -> pdf(y0_e,z))

y0_d = kde(df0.y)
plot(x, z -> pdf(y0_d,z))

#plot the sum of log earnings for school = 1
y1_e = kde(y1)
plot(x, z -> pdf(y1_e,z))

y1_d = kde(df1.y)
plot(x, z -> pdf(y1_d,z))





Xs = [Xs DataFrame(sel = repeat([0.],Ni))]

#this loop creates the difference of sum of future earnings in each choice
#note that xsb means x s[ummed]b, while xsbc is the counterfactual experience
#so, for school == 0 people, we use xsb, xsb2 for their β0, xsbc, xsbc2 for their β1
for i = 1:Ni
    if df.school[i] == 0
        Xs.sel[i] = Xs.xsa[i]*(β1[1]-β0[1]) + Xs.xsbc[i]β1[2] - Xs.xsb[i]*β0[2] + Xs.xsbc2[i]β1[3] - Xs.xsb2[i]*β0[3] #transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    else
        Xs.sel[i] = Xs.xsa[i]*(β1[1]-β0[1]) + Xs.xsb[i]β1[2] - Xs.xsbc[i]*β0[2] + Xs.xsb2[i]β1[3] - Xs.xsbc2[i]*β0[3]#transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
end

sel = Xs.sel .- [Xs.za Xs.zb Xs.zc Xs.zd Xs.ze]*δz .- θ.*(T - T*ρ - δt) .- rand(Normal(0,σ[9]),Ni)


#im getting the exact proportion. can't be.
mean(sel .> 0)
mean(Xs.school)
