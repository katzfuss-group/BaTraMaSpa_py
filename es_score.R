library(scoringRules)
rm(list = ls())


fn <- "true_and_sim.csv"
data <- as.matrix(read.csv2(fn, header = F, dec = ".", row.names = NULL, 
                            sep = ","))
cat("Vecchia NN sample energy score ", 
    es_sample(data[, 1], data[, 2 : ncol(data)]), "\n") # result 137.1049 

fn <- "true_and_sim.csv"
data <- as.matrix(read.csv2(fn, header = F, dec = ".", row.names = NULL, 
                            sep = ","))
cat("Deep learning model sample energy score ", 
    es_sample(data[, 1], data[, 2 : ncol(data)]), "\n") # result 93.46

fn <- "true_and_sim_R.csv"
data <- as.matrix(read.csv2(fn, header = F, dec = ".", row.names = NULL, 
                            sep = ","))
cat("GP sample energy score ", 
    es_sample(data[, 1], data[, 2 : ncol(data)]), "\n") # result 52.08921 

data.dummy <- data
data.dummy[, 2 : ncol(data)] <- mean(data[, 1])
cat("Dummy energy score ", 
    es_sample(data.dummy[, 1], data.dummy[, 2 : ncol(data.dummy)]), "\n") # result 72.62836
