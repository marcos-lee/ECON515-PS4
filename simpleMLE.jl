#this is a simpler version, where σ for wage equations don't vary with t
function mlesimplefake(theta::Array, df0::DataFrame, df1::DataFrame, Xs::DataFrame, quad::Array)
    #unpack parameters. Note that, to ensure that the variances are never below 0
    #the input is actually log(σ). Then, we exp(σ) to get the actual parameter
    β0 = theta[1:4]
    β1 = theta[5:8]
    σ = exp.(theta[13:16]) #the first T are σ0, one for each t. the last Ts are for σ1.
    δz = theta[9:10]
    δt = theta[11]
    σ0 = σ[1]
    σ1 = σ[2]
    σw = σ[3]
    σt = σ[4]
    ρ = theta[12]
    nnodes = quad[1]
    nodes = quad[2]
    weights = quad[3]
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
            temp0 = temp0 .* (pdf.(Normal(), (epsilon0s[df0.caseid .== i][j] .- ρ*sqrt(2)*σt.*nodes)./σ0))
        end
        integ0[i,:] = (temp0).*(1 .- cdf.(Normal(), (sel0[i] .- (T - T*ρ - δt)*sqrt(2)*σt.*nodes)./σw)).*weights
    end
    for i = 1:Ni1
        temp1 = (pdf.(Normal(), (epsilon1s[df1.caseid .== i][1] .- sqrt(2)*σt.*nodes)./σ1))
        for j = 2:T
            temp1 = temp1 .* (pdf.(Normal(), (epsilon1s[df1.caseid .== i][j] .- sqrt(2)*σt.*nodes)./σ1))
        end
        integ1[i,:] = (temp1).*(cdf.(Normal(), (sel1[i] .- (T - T*ρ - δt)*sqrt(2)*σt.*nodes)./σw)).*weights
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



#this is a simpler version, where σ for wage equations don't vary with t
function mlesimplenlsy(theta::Array, df0::DataFrame, df1::DataFrame, Xs::DataFrame)
    #unpack parameters. Note that, to ensure that the variances are never below 0
    #the input is actually log(σ). Then, we exp(σ) to get the actual parameter
    β0 = theta[1:3]
    β1 = theta[4:6]
    σ = exp.(theta[14:17]) #the first T are σ0, one for each t. the last Ts are for σ1.
    δz = theta[7:11]
    δt = theta[12]
    σ0 = σ[1]
    σ1 = σ[2]
    σw = σ[3]
    σt = σ[4]
    ρ = theta[13]
    #calculate the epsilons of the wage equation for each choice, for all (i,t), without θ
    epsilon0s = df0.y .- [df0.xa df0.xb df0.xb2]*β0
    epsilon1s = df1.y .- [df1.xa df1.xb df1.xb2]*β1
    #Xs is the individual-level dataframe, where all the X variables are summed, and Z is unique for each i.
    Xs = [Xs DataFrame(sel = repeat([0.],Ni))]
    Xs0 = Xs[Xs.school .== 0,:]
    Xs1 = Xs[Xs.school .== 1,:]
    #this loop creates the difference of sum of future earnings in each choice
    #note that xsb means x s[ummed]b, while xsbc is the counterfactual experience
    #so, for school == 0 people, we use xsb, xsb2 for their β0, xsbc, xsbc2 for their β1
    for i = 1:Ni0
        Xs0.sel[i] = Xs0.xsa[i]*(β1[1]-β0[1]) + Xs0.xsbc[i]β1[2] - Xs0.xsb[i]*β0[2] + Xs0.xsbc2[i]β1[3] - Xs0.xsb2[i]*β0[3] #transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    for i = 1:Ni1
        Xs1.sel[i] = Xs1.xsa[i]*(β1[1]-β0[1]) + Xs1.xsb[i]β1[2] - Xs1.xsbc[i]*β0[2] + Xs1.xsb2[i]β1[3] - Xs1.xsbc2[i]*β0[3]#transpose([Xs.xsa Xs.xsbc Xs.xsbc2 Xs.xsc][i,:])*β1 - transpose([Xs.xsa Xs.xsb Xs.xsb2 Xs.xsc][i,:])*β0
    end
    #this is the term inside the choice equation (without θ)
    sel0 = Xs0.sel .- [Xs0.za Xs0.zb Xs0.zc Xs0.zd Xs0.ze]*δz
    sel1 = Xs1.sel .- [Xs1.za Xs1.zb Xs1.zc Xs1.zd Xs1.ze]*δz
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
            temp0 = temp0 .* (pdf.(Normal(), (epsilon0s[df0.caseid .== i][j] .- ρ*sqrt(2)*σt.*nodes)./σ0))
        end
        integ0[i,:] = (temp0).*(1 .- cdf.(Normal(), (sel0[i] .- (T - T*ρ - δt)*sqrt(2)*σt.*nodes)./σw)).*weights
    end
    for i = 1:Ni1
        temp1 = (pdf.(Normal(), (epsilon1s[df1.caseid .== i][1] .- sqrt(2)*σt.*nodes)./σ1))
        for j = 2:T
            temp1 = temp1 .* (pdf.(Normal(), (epsilon1s[df1.caseid .== i][j] .- sqrt(2)*σt.*nodes)./σ1))
        end
        integ1[i,:] = (temp1).*(cdf.(Normal(), (sel1[i] .- (T - T*ρ - δt)*sqrt(2)*σt.*nodes)./σw)).*weights
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
