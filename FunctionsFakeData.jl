function mle1(theta::Array, df0::DataFrame, df1::DataFrame, Xs::DataFrame, quad::Array, constants::Array)
    #unpack parameters. Note that, to ensure that the variances are never below 0
    #the input is actually log(σ). Then, we exp(σ) to get the actual parameter
    T = constants[1]
    Ni = constants[2]
    Ni0 = constants[3]
    Ni1 = constants[4]
    β0 = theta[1:4]
    β1 = theta[5:8]
    σ = exp.(theta[13:13+2*T+1]) #the first T are σ0, one for each t. the last Ts are for σ1.
    δz = theta[9:10]
    δt = theta[11]
    σw = σ[2*T+1]
    σt = σ[2*T+2]
    ρ = theta[12]
    nnodes = quad[1]
    nodes = quad[2]
    weights = quad[3]
    #calculate the epsilons of the wage equation for each choice, for all (i,t), without θ
    epsilon0s = df0.y .- [df0.xa df0.xb df0.xb2 df0.xc]*β0
    epsilon1s = df1.y .- [df1.xa df1.xb df1.xb2 df1.xc]*β1
    #Xs is the individual-level dataframe, where all the X variables are summed, and Z is unique for each i.
    Xs0 = Xs[Xs.school .== 0,:]
    Xs1 = Xs[Xs.school .== 1,:]
    sel0 = Array{Any}(undef,Ni0)
    sel1 = Array{Any}(undef,Ni1)
    #this loop creates the difference of sum of future earnings in each choice
    #note that xsb means x s[ummed]b, while xsbc is the counterfactual experience
    #so, for school == 0 people, we use xsb, xsb2 for their β0, xsbc, xsbc2 for their β1
    for i = 1:Ni0
        sel0[i] = Xs0.xsa[i]*(β1[1]-β0[1]) + Xs0.xsbc[i]*β1[2] - Xs0.xsb[i]*β0[2] + Xs0.xsbc2[i]*β1[3] - Xs0.xsb2[i]*β0[3] + Xs0.xsc[i]*(β1[4]-β0[4]) #transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    for i = 1:Ni1
        sel1[i] = Xs1.xsa[i]*(β1[1]-β0[1]) + Xs1.xsb[i]*β1[2] - Xs1.xsbc[i]*β0[2] + Xs1.xsb2[i]*β1[3] - Xs1.xsbc2[i]*β0[3] + Xs1.xsc[i]*(β1[4]-β0[4])#transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    #this is the term inside the choice equation (without θ)
    sel0 = sel0 .- [Xs0.za Xs0.zb]*δz
    sel1 = sel1 .- [Xs1.za Xs1.zb]*δz
    #create empty matrices to store integration
    integ0 = Array{Any}(undef,Ni0,nnodes)
    integ1 = Array{Any}(undef,Ni1,nnodes)
    temp0 = Array{Any}(undef,nnodes,T)
    temp1 = Array{Any}(undef,nnodes,T)
    #this loop calculates for each i, the product (w.r.t time) of the pdfs of the wage eq. with the subtracted θ
    #the θs are actually the provided nodes of the gauss-hermite.
    #after it produces the product of pdfs, then I multiply the choice CDF probability times the weights of the quadrature
    #notice the change of variables necessary to do the approximation, ie, θ^2/2σ = x^2
    for i = 1:Ni0
        for j = 1:T
            temp0[:,j] = (pdf.(Normal(), (epsilon0s[df0.caseid .== i][j] .- ρ*sqrt(2)*σt.*nodes)./σ[j])./σ[j])
        end
        temp0_f = prod(temp0,dims=2)
        integ0[i,:] = (temp0_f).*(1 .- cdf.(Normal(), (sel0[i] .+ (T - T*ρ - δt)*sqrt(2)*σt.*nodes)./σw)).*weights
    end
    for i = 1:Ni1
        for j = 1:T
            temp1[:,j] = (pdf.(Normal(), (epsilon1s[df1.caseid .== i][j] .- sqrt(2)*σt.*nodes)./σ[T+j])./σ[T+j])
        end
        temp1_f = prod(temp1,dims=2)
        integ1[i,:] = (temp1_f).*(cdf.(Normal(), (sel1[i] .+(T - T*ρ - δt)*sqrt(2)*σt.*nodes)./σw)).*weights
    end
    #now, we have to sum across the number of nodes to obtain the numerical integration, that is, sum integ0/1 across rows
    contrib0 = Array{Any}(undef,Ni0)
    contrib1 = Array{Any}(undef,Ni1)
    for i = 1:Ni0
        contrib0[i] = sum(integ0[i,:])./sqrt(pi) #
    end
    for i = 1:Ni1
        contrib1[i] = sum(integ1[i,:])./sqrt(pi)
    end
    #so, contrib0 is the likelihood contribution of every individual i that has school == 0. now we take the log of each row and sum
    return ll = -sum(log.(contrib0)) - sum(log.(contrib1))
end
