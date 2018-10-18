# install.packages("readstata13")

library(readstata13)

setwd("C:/Users/Marcos Lee/Dropbox/Rice/Courses/Labor/PS4/")


fake <- read.dta13("fakedata_ps4.dta")

nlsy <- read.dta13("nlsy79_homework_data.dta")

write.csv(fake, file = "fakedata.csv")
write.csv(nlsy, file = "nlsy.csv")
