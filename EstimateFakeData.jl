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
df = load("fake_data_julia.csv") |> DataFrame #i changed the variable name income to y
#df = load("fake_data_julia_small.csv") |> DataFrame #I changed the variable name income to y

#define global number of periods and unique individuals
T = convert(Int64,maximum(df.age[df[:caseid] .== 1, :]) - minimum(df.age[df[:caseid] .== 1, :]) + 1)
Ni = convert(Int64,maximum(df.caseid))

#create period variable
t = repeat(1:T,Ni)
df = [df DataFrame(t=t) DataFrame(xbc = df.xb .- 4) ]

#create the counterfactual experience
df.xbc[df.school .== 1] = df.xbc[df.school .== 1] .+ 8
df = [df DataFrame(xb2 = df.xb.*df.xb) DataFrame(xbc2 = df.xbc.*df.xbc) ]

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
    DataFrame(xsa = sum(x.xa), xsb = sum(x.xb), xsb2 = sum(x.xb2), xsbc = sum(x.xbc), xsbc2 = sum(x.xbc2), xsc = sum(x.xc), za = mean(x.za), zb = mean(x.zb), school = mean(x.school))
end

#Number of unique individuals in each choice dataframe
Ni0 = convert(Int64,maximum(df0.caseid))
Ni1 = convert(Int64,maximum(df1.caseid))

constants = [T, Ni, Ni0, Ni1]


include("FunctionsFakeData.jl")

#I've coded up simple OLS and Probit estimators, to get sensible estimators
data0 = [df0.xa df0.xb df0.xb2 df0.xc]
data1 = [df1.xa df1.xb df1.xb2 df1.xc]

ols0 = OLS(df0.y, data0)
ols1 = OLS(df1.y, data1)

eps0 = log(sqrt(mean((df0.y .- data0 * ols0).^2)))
eps1 = log(sqrt(mean((df1.y .- data1 * ols1).^2)))

thetap = OLS(Xs.school, [ones(size(Xs.zb,1)) Xs.zb])

datap = [ones(size(Xs.zb,1)) Xs.zb Xs.school]

prob = TwiceDifferentiable(vars -> probit(vars, datap), thetap; autodiff = :forward)
@time prest = optimize(prob, thetap, BFGS(), Optim.Options(show_trace=true))


true_θ = [1.0, 2.0, -0.02,  0.5, 0.85, 3.5, -0.03, 1.0, 5.0, 3.0, 0.5, .8]
tσ0 = repeat([log(sqrt(0.25))], T)
tσ1 = repeat([log(sqrt(0.5))],T)
true_θ = vcat(true_θ, tσ0, tσ1, log(sqrt(.4)), log(1.))
#α_0,β_0b,γ_0b,β_0c,ρ_0,σ_0 = 1.0, 2.0,-0.02, 0.5, 0.8, sqrt(0.25)
#α_1,β_1b,γ_1b,β_1c,ρ_1,σ_1 = 0.85,3.5,-0.03,1.0,1.0,sqrt(0.5)
# primitives of distributions of unobservables
#σ_θ,σ_ω = sqrt(0.4),1.0
# primitive of cost function
#δ_a,δ_b,δ_θ = 5.0,3.0,0.5

#just guess actual parameters
β0 = [ols0[1], ols0[2], ols0[3], ols0[4]]
β1 = [ols1[1], ols1[2], ols1[3], ols1[4]]
σ0 = repeat([eps0],T)
σ1 = repeat([eps1],T)
σw = 0. #just use the guess from fake data
σt = log(sqrt(.4)) #just use the guess from fake data
δz = [-Optim.minimizer(prest)[1], -Optim.minimizer(prest)[2]]
δt = 0.5 #just use the guess from fake data
ρ = 0.8 #just use the guess from fake data

theta = vcat(β0, β1, δz, δt, ρ, σ0, σ1, σw, σt)
#define nodes and weights of gauss hermite
nnodes = 50
nodes, weight = gausshermite(nnodes)
quad = [nnodes, nodes, weight]

function wrapmle1(theta)
    return mle1(theta,df0,df1,Xs, quad, constants)
end

@time wrapmle1(theta)

func = TwiceDifferentiable(wrapmle1, theta; autodiff = :forward)
@time mini = optimize(func, theta, Newton(), Optim.Options(show_trace=true))

numerical_hessian = hessian!(func,Optim.minimizer(mini))
var_cov_matrix = inv(numerical_hessian)
diag(var_cov_matrix)


β0 = Optim.minimizer(mini)[1:4]
β1 = Optim.minimizer(mini)[5:8]
σ = exp.(Optim.minimizer(mini)[13:13+2*T+1])
σw = σ[2*T+1]
σt = σ[2*T+2]
δz = Optim.minimizer(mini)[9:10]
δt = Optim.minimizer(mini)[11]
ρ = Optim.minimizer(mini)[12]


θ = rand(Normal(0,σt),Ni)
θp = repeat(θ, inner=T)

ϵ0 = zeros(Ni*T)
ϵ1 = zeros(Ni*T)
for i = 1:Ni*T
    if df.t[i] == 1
        ϵ0[i] = rand(Normal(0,σ[1]))
    elseif df.t[i] == 2
        ϵ0[i] = rand(Normal(0,σ[2]))
    elseif df.t[i] == 3
        ϵ0[i] = rand(Normal(0,σ[3]))
    else
        ϵ0[i] = rand(Normal(0,σ[4]))
    end
end
for i = 1:Ni*T
    if df.t[i] == 1
        ϵ1[i] = rand(Normal(0,σ[5]))
    elseif df.t[i] == 2
        ϵ1[i] = rand(Normal(0,σ[6]))
    elseif df.t[i] == 3
        ϵ1[i] = rand(Normal(0,σ[7]))
    else
        ϵ1[i] = rand(Normal(0,σ[8]))
    end
end

y0 = zeros(Ni*T)
y1 = zeros(Ni*T)
for i = 1:Ni*T
    if df.school[i] == 1
        y0[i] = [df.xa df.xbc df.xbc2 df.xc][i,:]'*β0 .+ ρ.*θp[i] .+ ϵ0[i]
        y1[i] = [df.xa df.xb df.xb2 df.xc][i,:]'*β1 .+ θp[i] .+ ϵ1[i]
    else
        y0[i] = [df.xa df.xb df.xb2 df.xc][i,:]'*β0 .+ ρ.*θp[i] .+ ϵ0[i]
        y1[i] = [df.xa df.xbc df.xbc2 df.xc][i,:]'*β1 .+ θp[i] .+ ϵ1[i]
    end
end

#plot the sum of log earnings for school = 0
y0_e = kde(y0[df.school .== 0])
y0_d = kde(df0.y)
x = range(7, stop = 35, length = 250) |> collect
y0plot = [pdf(y0_e, x) pdf(y0_d, x)]
plot(x, y0plot,title="estimated density of Y0", label=["Simulated" "Data"])
savefig("tex/y0_fake")

#plot the sum of log earnings for school = 1
y1_e = kde(y1[df.school .== 1])
y1_d = kde(df1.y)
y1plot = [pdf(y1_e, x) pdf(y1_d, x)]
plot(x, y1plot,title="estimated density of Y1", label=["Simulated" "Data"])
savefig("tex/y1_fake")



Xs = [Xs DataFrame(sel = repeat([0.],Ni))]

#this loop creates the difference of sum of future earnings in each choice
#note that xsb means x s[ummed]b, while xsbc is the counterfactual experience
#so, for school == 0 people, we use xsb, xsb2 for their β0, xsbc, xsbc2 for their β1
for i = 1:Ni
    if Xs.school[i] == 0
        Xs.sel[i] = Xs.xsa[i]*(β1[1]-β0[1]) + Xs.xsbc[i]*β1[2] - Xs.xsb[i]*β0[2] + Xs.xsbc2[i]*β1[3] - Xs.xsb2[i]*β0[3] + Xs.xsc[i]*(β1[4]-β0[4])
    else
        Xs.sel[i] = Xs.xsa[i]*(β1[1]-β0[1]) + Xs.xsb[i]*β1[2] - Xs.xsbc[i]*β0[2] + Xs.xsb2[i]*β1[3] - Xs.xsbc2[i]*β0[3] + Xs.xsc[i]*(β1[4]-β0[4])
    end
end

ω = rand(Normal(0,σw),Ni)

sel = Xs.sel .- [Xs.za Xs.zb]*δz .+ θ.*(T - T*ρ - δt) .- ω



mean(sel .> 0)
mean(Xs.school)


summarystats(Xs.zb)
cutoff = mean(Xs.zb)
new_zb = zeros(Ni)

for i = 1:Ni
    if Xs.zb[i] < -.6284
        new_zb[i] = 0.
    else
        new_zb[i] = Xs.zb[i]
    end
end

Xs =[Xs DataFrame(zb_a = new_zb)]

sel_a = Xs.sel .- [Xs.za Xs.zb_a]*δz .+ θ.*(T - T*ρ - δt) .- ω


#im getting the exact proportion. can't be.
mean(sel_a .> 0)

lateind = (sel .> 0) .-(sel_a .> 0)

Ys = by(df, :caseid) do x
    DataFrame(ys1 = sum(exp.(y1)), ys0 = sum(exp.(y0)))
end

##estimate ATE
ATE = log(mean((Ys.ys1) .- (Ys.ys0)))

##estimate ATT
ATT = log(mean((Ys.ys1)[Xs.school .== 1] .- (Ys.ys0)[Xs.school .== 1]))

##estimate LATE


LATE = log(mean((Ys.ys1)[lateind .== 1] .- (Ys.ys0)[lateind .== 1]))





#I've coded up simple OLS and Probit estimators, to get sensible estimators
data0 = [df0.xa df0.xb df0.xb2 df0.xc]
data1 = [df1.xa df1.xb df1.xb2 df1.xc]

ols0 = OLS(df0.y, data0)
ols1 = OLS(df1.y, data1)

eps0 = log(sqrt(mean((df0.y .- data0 * ols0).^2)))
eps1 = log(sqrt(mean((df1.y .- data1 * ols1).^2)))

thetap = OLS(Xs.school, [ones(size(Xs.zb,1)) Xs.xsb Xs.xsb2 Xs.xsc Xs.zb])

datap = [ones(size(Xs.zb,1)) Xs.xsb Xs.xsb2 Xs.xsc Xs.zb Xs.school]

prob = TwiceDifferentiable(vars -> probit(vars, datap), thetap; autodiff = :forward)
@time prest = optimize(prob, thetap, BFGS(), Optim.Options(show_trace=true))
