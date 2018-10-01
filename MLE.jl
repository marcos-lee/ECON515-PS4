using Distributions
using LinearAlgebra
using Optim
using Statistics
using DataFrames
using CSVFiles
using FastGaussQuadrature

#df = load("fakedata.csv") |> DataFrame
df = load("fake_data_julia.csv") |> DataFrame
global T = convert(Int64,maximum(df.age[df[:caseid] .== 1, :]) - minimum(df.age[df[:caseid] .== 1, :]) + 1)
global Ni = convert(Int64,maximum(df.caseid))


t = repeat(1:T,20000)
df = [df DataFrame(t=t) DataFrame(xbc = df.xb .- 4) ]
df.xbc[df.school .== 1] = df.xbc[df.school .== 1] .+ 8
df = [df DataFrame(xb2 = df.xb.*df.xb) DataFrame(xbc2 = df.xbc.*df.xbc)]

dfsel = by(df, :caseid, x -> mean(x.school))

df0 = df[df.school .== 0,:]
sort!(df0, (:t, :caseid))
df0.caseid = repeat(1:size(df0[df0.t .== 1,:].caseid)[1],T)
sort!(df0, (:caseid, :t))

df1 = df[df.school .== 1,:]
sort!(df1, (:t, :caseid))
df1.caseid = repeat(1:size(df1[df1.t .== 1,:].caseid)[1],T)
sort!(df1, (:caseid, :t))

global nnodes = 5
global nodes, weights = gausshermite(nnodes)
Xs = by(df, :caseid) do x
    DataFrame(xsa = sum(x.xa), xsb = sum(x.xb), xsb2 = sum(x.xb2), xsbc = sum(x.xbc), xsbc2 = sum(x.xbc2), xsc = sum(x.xc), za = mean(x.za), zb = mean(x.zb), school = mean(x.school))
end

function mle(theta::Array, df::DataFrame, Xs::DataFrame)
    β0 = theta[1:4]
    β1 = theta[5:8]
    σ = theta[13:13+2*T+1]
    δz = theta[9:10]
    δt = theta[11]
    σ = exp.(σ)
    ρ = theta[12]
    epsilon0s = df.y .- [df.xa df.xb df.xb2 df.xc]*β0
    epsilon1s = df.y .- [df.xa df.xb df.xb2 df.xc]*β1
    #Xs = [Xs DataFrame(sel = repeat([transpose([Xs[:xsa] Xs[:xsbc] Xs[:xsbc2] Xs[:xsc]][1,:])*β1 - transpose([Xs[:xsa] Xs[:xsb] Xs[:xsb2] Xs[:xsc]][1,:])*β0],Ni))]
    Xs = [Xs DataFrame(sel = repeat([0.],Ni))]
    for i = 1:Ni
        if Xs[:school][i] == 1
            Xs.sel[i] = transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β0
        else
            Xs.sel[i] = transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
        end
    end
    Xs.sel = Xs.sel .- [Xs.za Xs.zb]*δz
    integ1 = zeros(size(Xs)[1],nnodes)
    integ0 = zeros(size(Xs)[1],nnodes)
    for i = 1:Ni
        temp1 = pdf.(Normal(), (epsilon1s[df.caseid .== i][1] .- sqrt(2)*σ[2*T+2].*nodes)./σ[T+1])./σ[T+1]
        temp0 = pdf.(Normal(), (epsilon0s[df.caseid .== i][1] .- ρ*sqrt(2)*σ[2*T+2].*nodes)./σ[1])./σ[1]
        for j = 2:T
            temp1 = temp1 .* pdf.(Normal(), (epsilon1s[df.caseid .== i][j] .- sqrt(2)*σ[2*T+2].*nodes)./σ[T+j])./σ[T+j]
            temp0 = temp0 .* pdf.(Normal(), (epsilon0s[df.caseid .== i][j] .- ρ*sqrt(2)*σ[2*T+2].*nodes)./σ[j])./σ[j]
        end
        integ1[i,:] = (temp1).*(cdf.(Normal(), (Xs.sel[i] .- sqrt(2)*σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights
        integ0[i,:] = (temp0).*(1 .- cdf.(Normal(), (Xs.sel[i] .- sqrt(2)*σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights
    end
    contrib0 = repeat([0.],size(Xs)[1])
    contrib1 = repeat([0.],size(Xs)[1])
    for i = 1:Ni
        contrib0[i] = sum(integ0[i,:])
        contrib1[i] = sum(integ1[i,:])
    end
    return ll = -sum(log.(contrib0[Xs.school .== 0])) - sum(log.(contrib1[Xs.school .==1]))
end

function mle1(theta::Array, df0::DataFrame, df1::DataFrame, Xs::DataFrame)
    β0 = theta[1:4]
    β1 = theta[5:8]
    σ = theta[13:13+2*T+1]
    δz = theta[9:10]
    δt = theta[11]
    σ = exp.(σ)
    ρ = theta[12]
    epsilon0s = df0.y .- [df0.xa df0.xb df0.xb2 df0.xc]*β0
    epsilon1s = df1.y .- [df1.xa df1.xb df1.xb2 df1.xc]*β1
    #Xs = [Xs DataFrame(sel = repeat([transpose([Xs[:xsa] Xs[:xsbc] Xs[:xsbc2] Xs[:xsc]][1,:])*β1 - transpose([Xs[:xsa] Xs[:xsb] Xs[:xsb2] Xs[:xsc]][1,:])*β0],Ni))]
    Xs = [Xs DataFrame(sel = repeat([0.],Ni))]
    for i = 1:Ni
        if Xs[:school][i] == 1
            Xs.sel[i] = transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β0
        else
            Xs.sel[i] = transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
        end
    end
    Xs.sel = Xs.sel .- [Xs.za Xs.zb]*δz
    sel0 = Xs.sel[Xs.school .== 0]
    sel1 = Xs.sel[Xs.school .== 1]
    Ni0 = convert(Int64,maximum(df0.caseid))
    Ni1 = convert(Int64,maximum(df1.caseid))
    integ0 = zeros(Ni0,nnodes)
    integ1 = zeros(Ni1,nnodes)
    for i = 1:Ni0
        temp0 = pdf.(Normal(), (epsilon0s[df0.caseid .== i][1] .- ρ*sqrt(2)*σ[2*T+2].*nodes)./σ[1])./σ[1]
        for j = 2:T
            temp0 = temp0 .* pdf.(Normal(), (epsilon0s[df0.caseid .== i][j] .- ρ*sqrt(2)*σ[2*T+2].*nodes)./σ[j])./σ[j]
        end
        integ0[i,:] = (temp0).*(1 .- cdf.(Normal(), (sel0[i] .- sqrt(2)*σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights
    end
    for i = 1:Ni1
        temp1 = pdf.(Normal(), (epsilon1s[df1.caseid .== i][1] .- sqrt(2)*σ[2*T+2].*nodes)./σ[T+1])./σ[T+1]
        for j = 2:T
            temp1 = temp1 .* pdf.(Normal(), (epsilon1s[df1.caseid .== i][j] .- sqrt(2)*σ[2*T+2].*nodes)./σ[T+j])./σ[T+j]
        end
        integ1[i,:] = (temp1).*(cdf.(Normal(), (sel1[i] .- sqrt(2)*σ[2*T+2].*nodes.*δt) ./ σ[2*T+1])).*1/(σ[2*T+2]*sqrt(2*pi)).*weights
    end
    contrib0 = repeat([0.],Ni0)
    contrib1 = repeat([0.],Ni1)
    for i = 1:Ni0
        contrib0[i] = sum(integ0[i,:])
    end
    for i = 1:Ni1
        contrib1[i] = sum(integ1[i,:])
    end
    return ll = -sum(log.(contrib0)) - sum(log.(contrib1))
end


β0 = [1. 2. -0.01 0.5]
β1 = [.9 3.4 -0.01 1.]
σ = [transpose(repeat([log(sqrt(0.25))],T)) transpose(repeat([log(sqrt(0.5))],T)) 1. log(sqrt(.4))]
δz = [5.1 3.1]
δt = 0.5
ρ = 0.8

theta = [β0 β1 δz δt ρ σ]

@time mle(theta,df, Xs)
@time mle1(theta,df0, df1, Xs)



@time mini = optimize(vars -> mle1(vars, df0, df1, Xs), theta)


@time mini1 = optimize(vars -> mle1(vars, df0, df1, Xs), theta, BFGS())



#=
Optim.minimizer(mini)[11:36] = exp.(Optim.minimizer(mini)[11:36]).^2


β0 = Optim.minimizer(mini)[1:3]
β1 = Optim.minimizer(mini)[4:6]
σ = Optim.minimizer(mini)[11:36]
δz = Optim.minimizer(mini)[7:8]
δt = Optim.minimizer(mini)[9]
ρ = Optim.minimizer(mini)[10]


σ = exp.(σ)
=#
