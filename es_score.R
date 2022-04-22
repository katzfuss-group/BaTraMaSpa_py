library(scoringRules)
rm(list = ls())


fn <- "true_and_sim.csv"
data <- as.matrix(read.csv2(fn, header = F, dec = ".", row.names = NULL, 
                            sep = ","))
cat("Vecchia NN sample energy score ", 
    es_sample(data[, 1], data[, 2 : ncol(data)]), "\n") # result 137.1049 



