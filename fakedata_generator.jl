# Sen Lu, Labor Economics, Problem set 4
# This data generate process is exactly the same as Dr. Cunha 's stata do file
using Distributions
using DataFrames
using CSV

# Model Primitives and notations:
#=
This script aim to estimate primitive using fake data.
=#

#--- set working directory:
cd("C:\\working\\cs\\Econometrics\\511_Julia\\LaborEcon\\ps4")

#--- Set number of observations:
N = 20000;
indx = [i for i=1:N]

#--- Set up primitives of parameters of interest:
# primitives in income function
α_0,β_0b,γ_0b,β_0c,ρ_0,σ_0 = 1.0, 2.0,-0.02, 0.5, 0.8, sqrt(0.25)
α_1,β_1b,γ_1b,β_1c,ρ_1,σ_1 = 0.85,3.5,-0.03,1.0,1.0,sqrt(0.5)
# primitives of distributions of unobservables
σ_θ,σ_ω = sqrt(0.4),1.0
# primitive of cost function
δ_a,δ_b,δ_θ = 5.0,3.0,0.5

#--- Generate Covariate
xa = ones(N,4)     #intercept
xb1 = zeros(N,4)
for i=1:4
    xb1[:,i] = 3+i  #experience if school
end
xb0 = zeros(N,4)
for i=1:4
    xb0[:,i] = 7+i  #experience if not school
end
xc = randn(N)      #observable heterogeneity
za = ones(N)        #intercept
zb = randn(N)       #exclusion restriction

# Unobservables for econometricians
θ = σ_θ * randn(N)  # Unobservable heterogeneity
ϵ_0 = σ_0 * randn(N,4)  # income shocks if not school
ϵ_1 = σ_1 * randn(N,4)  # income shocks if school
ω = σ_ω * randn(N)      # unobservable

#--- Define necessary functions:
function EY_year(Sch::Bool, indx::Int64, year::Int64)
    EY_year_cache = 0.0
    if Sch
        EY_year_cache += α_1 + β_1b * xb1[indx,year] + γ_1b * (xb1[indx,year]^2) + β_1c * xc[indx] + ρ_1 * θ[indx]
    else
        EY_year_cache += α_0 + β_0b * xb0[indx,year] + γ_0b * (xb0[indx,year]^2) + β_0c * xc[indx] + ρ_0 * θ[indx]
    end
    return EY_year_cache
end

function EY(Sch::Bool, indx::Int64)
    EY_cache = 0.0
    if Sch
        for i=1:4
            EY_cache += α_1 + β_1b * xb1[indx,i] + γ_1b * (xb1[indx,i]^2) + β_1c * xc[indx] + ρ_1 * θ[indx]
        end
    else
        for i=1:4
            EY_cache += α_0 + β_0b * xb0[indx,i] + γ_0b * (xb0[indx,i]^2) + β_0c * xc[indx] + ρ_0 * θ[indx]
        end
    end
    return EY_cache
end


function SchCost(indx::Int64)
    Cost_cache = 0.0
    Cost_cache = δ_a*za[indx] + δ_b*zb[indx] + δ_θ*θ[indx] + ω[indx]
    return Cost_cache
end


function SchChoice(indx::Int64)
    criterion = EY(true,indx) - EY(false,indx) - SchCost(indx)
    if criterion >= 0
        return true
    else
        return false
    end
end

#--- Use functions defined in last section to simulate schooling choice
Sch =[SchChoice(i) for i=1:N]
# necessary to check the observed schooling rate:
#mean(Sch)

#---- Use functions to simulate income outcome
Y_0 = similar(ϵ_0)
Y_1 = similar(ϵ_1)

for i=1:N
    Y_0[i,:] = [EY_year(false,i,j)+ϵ_0[i,j] for j =1:4]
end

for i=1:N
    Y_1[i,:] = [EY_year(true,i,j)+ϵ_1[i,j] for j =1:4]
end

age = Array{Int64}(N,4)
for i=1:N
    age[i,:] =[j for j=26:29]
end

#--- merge data
function case_data_year(case_id::Int64, year::Int64)
    data_cache = []
    caseid_data = case_id
    age_data = age[case_id,year]
    schooling_data = Sch[case_id]
    if schooling_data
        income_data = Y_1[case_id,year]
        xb_data = xb1[case_id,year]
    else
        income_data = Y_0[case_id,year]
        xb_data = xb0[case_id,year]
    end
    xa_data = xa[case_id]
    xc_data = xc[case_id]
    za_data = za[case_id]
    zb_data = zb[case_id]
    data_cache = [caseid_data age_data schooling_data income_data xa_data xb_data xc_data za_data zb_data]
    return data_cache
end

function case_data(case_id::Int64)
    # data should include: caseid;age;schooling;income;xa;xb;za;zb
    data_cache = []
    for year =1:4
        data_cache = vcat(data_cache,case_data_year(case_id,year))
    end
    return data_cache
end

# Create an array to store the data
data =Array{Float64, 2}(4*N,size(case_data_year(1,1),2))
for i=1:N
    data[4*(i-1)+1:4*i,:] = case_data(i)
end

#--- Dataframes
df = DataFrame(data)
names!(df, [Symbol("caseid"),Symbol("age"),Symbol("school"),Symbol("income"),Symbol("xa"),Symbol("xb"),Symbol("xc"),Symbol("za"),Symbol("zb")])
describe(df)
CSV.write("fake_data_julia.csv",df)
