clear all
set mem 300m

cd D:\fcunha\Dropbox\Teaching\Rice\Graduate_Labor_Economics\Fall2018\ProblemSets

set obs 20000
gen caseid = _n

gen lambda = sqrt(0.4)*invnorm(uniform())

gen xa = 1
gen xb = 0.0
gen xc = 1.0*invnorm(uniform())
gen EV0 = 0.0
gen EV1 = 0.0

local age0 = 4
local age1 = 8

local t = `age0'
	while `t' < `age1' {
	replace xb = `t' + 4
	gen EY0`t' = 1.00*xa + 2.0*xb - 0.02*xb*xb + 0.5*xc + 0.8*lambda
	replace EV0 = EV0 + EY0`t'
	replace xb = `t'
	gen EY1`t' = 0.85*xa + 3.5*xb - 0.03*xb*xb + 1.0*xc + 1.0*lambda
	replace EV1 = EV1 + EY1`t'
	local t = `t' + 1
}

gen za = 1.0
gen zb = 1.0*invnorm(uniform())
gen C  = 5.0*za + 3.0*zb + 0.5*lambda + sqrt(1.0)*invnorm(uniform())
gen V = EV1-EV0-C
gen school = 0
replace school = 1 if V >= 0

local t = `age0'
	while `t' < `age1' {
	gen Y0`t' = EY0`t' + sqrt(0.25)*invnorm(uniform())
	gen Y1`t' = EY1`t' + sqrt(0.50)*invnorm(uniform())
	gen Y_`t'  = (1.0d0-school)*Y0`t' + school*Y1`t'
	local t = `t' + 1
}

drop lambda EY0* EY1*  C V Y0* Y1* EV0 EV1

local t = `age0'
	while `t' < `age1' {
	ren Y_`t' Y`t'
	local t = `t' + 1
}

reshape long Y, i(caseid) j(age)
replace xb = age + 4 if school == 0 
replace xb = age if school == 1
replace age = age + 22

label variable xb "potential experience"
label variable xa "intercept"
label variable za "intercept"
label variable zb "exclusion restriction"
label variable Y "total income from wages and salary"
label variable school "educational attainment, 1=college, 0=hs"

ren Y y

order caseid age school y xa xb za zb

sort caseid age
save fakedata_ps4, replace
