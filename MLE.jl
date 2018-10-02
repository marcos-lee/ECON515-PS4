using Distributions
using LinearAlgebra
using Optim
using Statistics
using DataFrames
using CSVFiles
using FastGaussQuadrature
using NLSolversBase

#df = load("fakedata.csv") |> DataFrame
df = load("fake_data_julia_small.csv") |> DataFrame #i changed the variable name income to y

#define global number of periods and unique individuals
const T = convert(Int64,maximum(df.age[df[:caseid] .== 1, :]) - minimum(df.age[df[:caseid] .== 1, :]) + 1)
const Ni = convert(Int64,maximum(df.caseid))

#create period variable
t = repeat(1:T,Ni)
df = [df DataFrame(t=t) DataFrame(xbc = df.xb .- 4) ]

#create the counterfactual experience
df.xbc[df.school .== 1] = df.xbc[df.school .== 1] .+ 8
df = [df DataFrame(xb2 = df.xb.*df.xb) DataFrame(xbc2 = df.xbc.*df.xbc)]


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
const Ni0 = convert(Int64,maximum(df0.caseid))
const Ni1 = convert(Int64,maximum(df1.caseid))

#define nodes and weights of gauss hermite
const nnodes = 5
const nodes, weights = gausshermite(nnodes)


function mle1(theta::Array, df0::DataFrame, df1::DataFrame, Xs::DataFrame)
    #unpack parameters. Note that, to ensure that the variances are never below 0
    #the input is actually log(σ). Then, we exp(σ) to get the actual parameter
    β0 = theta[1:4]
    β1 = theta[5:8]
    σ = theta[13:13+2*T+1] #the first T are σ0, one for each t. the last Ts are for σ1.
    δz = theta[9:10]
    δt = theta[11]
    σ = exp.(σ)
    σw = σ[2*T+1]
    σt = σ[2*T+2]
    ρ = theta[12]
    #calculate the epsilons of the wage equation for each choice, for all (i,t), without θ
    epsilon0s = df0.y .- [df0.xa df0.xb df0.xb2 df0.xc]*β0
    epsilon1s = df1.y .- [df1.xa df1.xb df1.xb2 df1.xc]*β1
    #Xs is the individual-level dataframe, where all the X variables are summed, and Z is unique for each i.
    Xs = [Xs DataFrame(sel = repeat([0.],Ni))]
    Xs0 = Xs[Xs.school .== 0,:]
    Xs1 = Xs[Xs.school .== 1,:]
    #this loop creates the difference of sum of future earnings in each choice
    #note that xsb means x s[ummed]b, while xsbc is the counterfactual experience
    #so, for school == 0 people, we use xsb, xsb2 for their β0, xsbc, xsbc2 for their β1
    for i = 1:Ni0
        Xs0.sel[i] = Xs0.xsa[i]*(β1[1]-β0[1]) + Xs0.xsbc[i]β1[2] - Xs0.xsb[i]*β0[2] + Xs0.xsbc2[i]β1[3] - Xs0.xsb2[i]*β0[3] + Xs0.xsc[i]*(β1[4]-β0[4]) #transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    for i = 1:Ni1
        Xs1.sel[i] = Xs1.xsa[i]*(β1[1]-β0[1]) + Xs1.xsb[i]β1[2] - Xs1.xsbc[i]*β0[2] + Xs1.xsb2[i]β1[3] - Xs1.xsbc2[i]*β0[3] + Xs1.xsc[i]*(β1[4]-β0[4])#transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    #this is the term inside the choice equation (without θ)
    sel0 = Xs0.sel .- [Xs0.za Xs0.zb]*δz
    sel1 = Xs1.sel .- [Xs1.za Xs1.zb]*δz
    #create empty matrices to store integration
    integ0 = zeros(Ni0,nnodes)
    integ1 = zeros(Ni1,nnodes)
    #this loop calculates for each i, the product (w.r.t time) of the pdfs of the wage eq. with the subtracted θ
    #the θs are actually the provided nodes of the gauss-hermite.
    #after it produces the product of pdfs, then I multiply the choice CDF probability times the weights of the quadrature
    #notice the change of variables necessary to do the approximation, ie, θ^2/2σ = x^2
    for i = 1:Ni0
        temp0 = (pdf.(Normal(), (epsilon0s[df0.caseid .== i][1] .- ρ*sqrt(2)*σt.*nodes)./σ[1])./σ[1])
        for j = 2:T
            temp0 = temp0 .* (pdf.(Normal(), (epsilon0s[df0.caseid .== i][1] .- ρ*sqrt(2)*σt.*nodes)./σ[j])./σ[j])
        end
        integ0[i,:] = (temp0).*(1 .- cdf.(Normal(), (sel0[i] .- δt *sqrt(2)*σt.*nodes)./σw)).*weights
    end
    for i = 1:Ni1
        temp1 = (pdf.(Normal(), (epsilon1s[df1.caseid .== i][1] .- sqrt(2)*σt.*nodes)./σ[T+1])./σ[T+1])
        for j = 2:T
            temp1 = temp1 .* (pdf.(Normal(), (epsilon1s[df1.caseid .== i][1] .- sqrt(2)*σt.*nodes)./σ[T+j])./σ[T+j])
        end
        integ1[i,:] = (temp1).*(cdf.(Normal(), (sel1[i] .- δt *sqrt(2)*σt.*nodes)./σw)).*weights
    end
    #now, we have to sum across the number of nodes, that is, sum integ0/1 across rows
    contrib0 = repeat([0.],Ni0)
    contrib1 = repeat([0.],Ni1)
    for i = 1:Ni0
        contrib0[i] = sum(integ0[i,:])./sqrt(pi) #
    end
    for i = 1:Ni1
        contrib1[i] = sum(integ1[i,:])./sqrt(pi)
    end
    #so, contrib0 is the likelihood contribution of every individual i. now we take the log of each row and sum
    return ll = -sum(log.(contrib0)) - sum(log.(contrib1))
end

#this is a simpler version, where σ for wage equations don't vary with t
function mlesimple(theta::Array, df0::DataFrame, df1::DataFrame, Xs::DataFrame)
    #unpack parameters. Note that, to ensure that the variances are never below 0
    #the input is actually log(σ). Then, we exp(σ) to get the actual parameter
    β0 = theta[1:4]
    β1 = theta[5:8]
    σ = theta[13:16] #the first T are σ0, one for each t. the last Ts are for σ1.
    δz = theta[9:10]
    δt = theta[11]
    σ = exp.(σ)
    σ0 = σ[1]
    σ1 = σ[2]
    σw = σ[3]
    σt = σ[4]
    ρ = theta[12]
    #calculate the epsilons of the wage equation for each choice, for all (i,t), without θ
    epsilon0s = df0.y .- [df0.xa df0.xb df0.xb2 df0.xc]*β0
    epsilon1s = df1.y .- [df1.xa df1.xb df1.xb2 df1.xc]*β1
    #Xs is the individual-level dataframe, where all the X variables are summed, and Z is unique for each i.
    Xs = [Xs DataFrame(sel = repeat([0.],Ni))]
    Xs0 = Xs[Xs.school .== 0,:]
    Xs1 = Xs[Xs.school .== 1,:]
    #this loop creates the difference of sum of future earnings in each choice
    #note that xsb means x s[ummed]b, while xsbc is the counterfactual experience
    #so, for school == 0 people, we use xsb, xsb2 for their β0, xsbc, xsbc2 for their β1
    for i = 1:Ni0
        Xs0.sel[i] = Xs0.xsa[i]*(β1[1]-β0[1]) + Xs0.xsbc[i]β1[2] - Xs0.xsb[i]*β0[2] + Xs0.xsbc2[i]β1[3] - Xs0.xsb2[i]*β0[3] + Xs0.xsc[i]*(β1[4]-β0[4]) #transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    for i = 1:Ni1
        Xs1.sel[i] = Xs1.xsa[i]*(β1[1]-β0[1]) + Xs1.xsb[i]β1[2] - Xs1.xsbc[i]*β0[2] + Xs1.xsb2[i]β1[3] - Xs1.xsbc2[i]*β0[3] + Xs1.xsc[i]*(β1[4]-β0[4])#transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    #this is the term inside the choice equation (without θ)
    sel0 = Xs0.sel .- [Xs0.za Xs0.zb]*δz
    sel1 = Xs1.sel .- [Xs1.za Xs1.zb]*δz
    #create empty matrices to store integration
    integ0 = zeros(Ni0,nnodes)
    integ1 = zeros(Ni1,nnodes)
    #this loop calculates for each i, the product (w.r.t time) of the pdfs of the wage eq. with the subtracted θ
    #the θs are actually the provided nodes of the gauss-hermite.
    #after it produces the product of pdfs, then I multiply the choice CDF probability times the weights of the quadrature
    #notice the change of variables necessary to do the approximation, ie, θ^2/2σ = x^2
    for i = 1:Ni0
        temp0 = (pdf.(Normal(), (epsilon0s[df0.caseid .== i][1] .- ρ*sqrt(2)*σt.*nodes)./σ0))
        for j = 2:T
            temp0 = temp0 .* (pdf.(Normal(), (epsilon0s[df0.caseid .== i][1] .- ρ*sqrt(2)*σt.*nodes)./σ0))
        end
        integ0[i,:] = (temp0).*(1 .- cdf.(Normal(), (sel0[i] .- δt *sqrt(2)*σt.*nodes)./σw)).*weights
    end
    for i = 1:Ni1
        temp1 = (pdf.(Normal(), (epsilon1s[df1.caseid .== i][1] .- sqrt(2)*σt.*nodes)./σ1))
        for j = 2:T
            temp1 = temp1 .* (pdf.(Normal(), (epsilon1s[df1.caseid .== i][1] .- sqrt(2)*σt.*nodes)./σ1))
        end
        integ1[i,:] = (temp1).*(cdf.(Normal(), (sel1[i] .- δt *sqrt(2)*σt.*nodes)./σw)).*weights
    end
    #now, we have to sum across the number of nodes, that is, sum integ0/1 across rows
    contrib0 = repeat([0.],Ni0)
    contrib1 = repeat([0.],Ni1)
    for i = 1:Ni0
        contrib0[i] = sum(integ0[i,:])./(sqrt(pi)*σ0) #
    end
    for i = 1:Ni1
        contrib1[i] = sum(integ1[i,:])./(sqrt(pi)*σ1)
    end
    #so, contrib0 is the likelihood contribution of every individual i. now we take the log of each row and sum
    return ll = -sum(log.(contrib0)) - sum(log.(contrib1))


end



#guesses are basically the true parameters.
β0 = [1., 2., -0.01, 0.5]
β1 = [.85, 3.4, -0.03, 1.]
σ0 = repeat([log(sqrt(0.25))], T)
σ1 = repeat([log(sqrt(0.5))],T)
σw = 1.
σt = log(sqrt(.4))
δz = [5., 3.1]
δt = 0.5
ρ = 0.8

theta = vcat(β0, β1, δz, δt, ρ, σ0, σ1, σw, σt)


@time mle1(theta,df0, df1, Xs)


#optimization procedure
@time mini = optimize(vars -> mle1(vars, df0, df1, Xs), theta, Optim.Options(iterations = 10000))
#had to increase max iterations, now it convergers, but results are way off


#now estimate using simpler model
β0 = [1., 1., 0.01, 0.1]
β1 = [1., 2., 0.01, 1.]
σ0 = log(sqrt(0.25))
σ1 = log(sqrt(0.5))
σw = 1.
σt = log(sqrt(.4))
δz = [5., 3.1]
δt = 0.5
ρ = 0.8

theta = vcat(β0, β1, δz, δt, ρ, σ0, σ1, σw, σt)
@time mlesimple(theta,df0, df1, Xs)

@time mini1 = optimize(vars -> mlesimple(vars, df0, df1, Xs), theta, Optim.Options(iterations = 10000))
#had to increase max iterations, now it convergers, but results are way off


#its taking 4+ hours and reaching max iteration. therefore, it must be wrong

#@time mini1 = optimize(vars -> mle1(vars, df0, df1, Xs), theta, BFGS())



#=
Optim.minimizer(mini)[13:13+2*T+1] = exp.(Optim.minimizer(mini)[13:13+2*T+1]).^2

β0 = Optim.minimizer(mini)[1:4]
β1 = Optim.minimizer(mini)[5:8]
σ = Optim.minimizer(mini)[13:13+2*T+1]
δz = Optim.minimizer(mini)[9:10]
δt = Optim.minimizer(mini)[11]
ρ = Optim.minimizer(mini)[12]


σ = exp.(σ)
=#

