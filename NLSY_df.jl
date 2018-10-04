using Distributions
using LinearAlgebra
using Optim
using Statistics
using DataFrames


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
