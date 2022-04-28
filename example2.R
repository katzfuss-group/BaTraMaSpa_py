devtools::install_github("https://github.com/katzfuss-group/GpGp.git",
                         ref = "squared_relevance")
library(GpGp)
library(GPvecchia)
library(tidyverse)
library(parallel)
library(scoringRules)
rm(list = ls())
set.seed(1)
cluster <- makeCluster(min(detectCores() - 1, 16))

# read data
data <- read_csv("data/prec.csv", col_names = FALSE)
n <- ncol(data) - 1
ns <- nrow(data) - 3
data <- data[, 2 : (n + 1)]
locs <- t(as.matrix(data[1 : 3, ]))
y <- as.matrix(data[4 : (ns + 3), ])
rm(data)

# maxmin and NN
odr <- order_maxmin(locs)
locs <- locs[odr, ]
y <- y[, odr]
m <- 30
NN <- find_ordered_nn(locs, m)

# Vecchia MLE
linkfuns <- GpGp::get_linkfun("matern25_scaledim")
link <- linkfuns$link
dlink <- linkfuns$dlink
invlink <- linkfuns$invlink
obj_func <- function(parms.trans){
    parms <- link(parms.trans)
    loglik <- 0
    grad <- rep(0, length(parms))
    info <- matrix(0, length(parms), length(parms))
    clusterExport(cluster, varlist = c("NN", "locs", "parms"), 
                  envir = environment())
    tmp_func <- function(y){
        GpGp::vecchia_meanzero_loglik_grad_info(parms, "matern25_scaledim", y, 
                                                locs, NN)
    }
    rslt <- parApply(cluster, y[1 : (ns - 1), , drop = F], MARGIN = 1, 
                     FUN = tmp_func)
    for(i in 1 : (ns - 1)){
        loglik = loglik + rslt[[i]]$loglik
        grad = grad + rslt[[i]]$grad
        info = info + rslt[[i]]$info
    }
    loglik <- - loglik
    grad <- - grad * dlink(parms.trans)
    info <- info * outer(dlink(parms.trans), dlink(parms.trans))
    return(list(loglik = loglik, grad = grad, info = info))
}
parms0 <- c(1, 1, 1, 1, 0.1)
parms0.trans <- invlink(parms0)
FisherObj <- fisher_scoring_meanzero(obj_func, parms0.trans, link, F, 1e-4, 20)
parms.trans <- FisherObj$logparms
parms <- link(parms.trans)

# posterior sampling
n.known <- floor(n * 0.1)
samp <- cond_sim(locs_pred = locs[(1 + n.known) : n, ], 
                 X_pred = matrix(0, n - n.known, 1), y_obs = y[ns, 1 : n.known], 
                 locs_obs = locs[1 : n.known, ], X_obs = matrix(0, n.known, 1), 
                 beta = 0, covparms = parms, covfun_name = "matern25_scaledim", 
                 m = m, nsims = 1000)
samp <- rbind(matrix(y[ns, 1 : n.known], n.known, ncol(samp), byrow = F), samp)
cat("GP posterior sample energy score ", 
    es_sample(y[ns, ], samp), "\n") # result 65.85292 
i <- 1
samp.i <- samp[, i]
samp.i[odr] <- samp.i
image(matrix(samp.i, 288, 192, byrow = F))

# Posterior mean
n.known <- floor(n * 0.1)
y.pred <- predictions(locs_pred = locs[(1 + n.known) : n, ], 
                      X_pred = matrix(0, n - n.known, 1), y_obs = y[ns, 1 : n.known], 
                      locs_obs = locs[1 : n.known, ], X_obs = matrix(0, n.known, 1), 
                      beta = 0, covparms = parms, covfun_name = "matern25_scaledim", 
                      m = m)
y.pred <- c(y[ns, 1 : n.known], y.pred) # 0.3941549
sqrt(mean((y.pred - y[ns,])^2))
i <- 1
samp.i <- samp[, i]
samp.i[odr] <- samp.i
image(matrix(samp.i, 288, 192, byrow = F))





