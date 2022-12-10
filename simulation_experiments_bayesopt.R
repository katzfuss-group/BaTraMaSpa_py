setwd("C://PhD Research//Self//ClimModCalibration//BaTraMaSpa_py")
library(ggplot2)
##univarite functions optimization---------------------------------------------------

#defining f1, f2, and log-score
f1 = function(x) cos(x)
f2 = function(x) sin(x)

log_score_fun = function(y, mu, sigmasq) 
{return(sum(0.5 * log(sigmasq) + (((y-mu)^2)/(2*sigmasq))))}

kldiv = function(x, xstar, ystar)
{
  n = length(ystar)
  klxstar = -(n * f2(xstar)/2) - n/2
  klx = -(n * f2(x)/2) - (n/2) * exp(f2(xstar) - f2(x)) -
    (n/2) * ((f1(x) - f1(xstar))^2) * exp(-f2(x))
  div = klxstar - klx
  return(div)
}

#https://stackoverflow.com/questions/61589781/plot-vertical-density-of-normal-distribution-in-r-and-ggplot2

#plot of f1(x) and f2(x) (including specification of xstar and ystar)
set.seed(1234); xstar = 1.5; N = 10
ystar = f1(xstar) + sqrt(exp(f2(xstar))) * rnorm(N)
x_plot = c(-2.7, -1.5, 0, 1.5, 2.7); sd = sqrt(exp(sin(x_plot)))

k = 3

base_plot_f1 <- ggplot(data = data.frame(x = c(-4, 4), y = c(-6, 6)), aes(x, y)) +
  geom_function(fun = cos, color = "red") + 
  geom_function(fun = function(x) cos(x) - 1.96 * sqrt(exp(sin(x))), color = "blue") +
  geom_function(fun = function(x) cos(x) + 1.96 * sqrt(exp(sin(x))), color = "blue") 
p1<- base_plot_f1 +
  ylab("") + coord_cartesian(ylim = c(-6, 6)) + ggtitle("Plot of cos(x) and 95% confidence band")
for(i in 1:length(x_plot)){
  
  x = seq( - k * sd[i], k * sd[i], length.out = 60)
  y = dnorm(x, 0, sd[i])#/dnorm(0, 0, sd[i]) 
  path1 = data.frame(x = -y + x_plot[i], y = x + f1(x_plot[i]))
  segment1 = data.frame(x = x_plot[i], y = cos(x_plot[i]) - k * sd[i],
                        xend =  x_plot[i], yend = cos(x_plot[i]) + k * sd[i])
  p1 = p1 +
    geom_path(aes(x, y), data = path1, color = "green") + 
    geom_segment(aes(x = x, y = y, xend = xend, yend = yend), data = segment1)
}
p1 <- p1 + scale_y_continuous(breaks = NULL)
#p1

p2 <- ggplot(data = data.frame(x = c(-4, 4), y = c(-6, 6)), aes(x, y)) +
  geom_function(fun = function(x) sqrt(exp(sin(x))), color = "green") +
  ylab("") + coord_cartesian(ylim = c(0, 2)) + scale_y_continuous(breaks = NULL) +
  ggtitle("Plot of f2(x)")
#p2

x_grid_calc = seq(-4, 4, by = 0.1) #grid for calculation of loglikelihoods at x
empll = c()
for(h in x_grid_calc){
  empll = append(empll, 
                 ((N/2) * f2(h) + sum(((ystar - f1(h))^2) * exp(-f2(h)))/2))
}
p3 <- ggplot(data = data.frame(x = c(-4, 4), y = c(500, 5000)), aes(x, y)) +
  geom_function(fun = kldiv, args = list(xstar = xstar, ystar = ystar), color = "red") +
  geom_point(data = data.frame(x = x_grid_calc, y = empll), col = "green") +
  geom_hline(yintercept = ((N/2) * f2(xstar) + sum(((ystar - f1(xstar))^2) * exp(-f2(xstar)))/2),
             col = "green", linetype = "dashed") +
  ylab("")

#choosing grid
gridsize = 100
x_grid = seq(-4, 4, length.out = gridsize)

#simulating approximating prior variance for f2
var_f2_error = rep(0, length(x_grid)); mean_f2_error = rep(0, length(x_grid))
var_f1 = (exp(f2(x_grid)))/ N #empirical variance for the data
for(i in 1:(length(x_grid))){
  
  var_hstry = c()
  for(j in 1:10000){
    
    y_temp = rnorm(N, mean = f1(x_grid[i]), sd = sqrt(exp(f2(x_grid[i])))) 
    var_hstry = append(var_hstry, log(var(y_temp)))
  }
  var_hstry = var_hstry - f2(x_grid[i])
  mean_f2_error[i] = mean(var_hstry)
  var_f2_error[i] = var(var_hstry)
  remove(var_hstry)
}

#GP prior structure
l1 = 1.2; l2 = 1.2;
cov_matrix_f1 = exp( - (fields::rdist(x_grid))^2/ l1^2) + diag(10^-10, length(x_grid))
cov_matrix_f2 = exp( - (fields::rdist(x_grid))^2/ l2^2) + diag(10^-10, length(x_grid))

#primary prediction check to see everything is working
rndptindx = c(30, 60, 90)
ystarrnd = matrix(0, ncol = 3, nrow = N)
for(j in 1:3)
{
  ystarrnd[ ,j] = f1(x_grid[rndptindx[j]]) + sqrt(exp(f2(x_grid[rndptindx[j]]))) * rnorm(N)
}
hstry_mean_rndpt = apply(ystarrnd, 2, mean); hstry_var_rndpt = apply(ystarrnd, 2, var)
hstry_var_rndpt = log(hstry_var_rndpt)

mean_f2_post = rep(0, 100)

postvar_f1 = solve(cov_matrix_f1[rndptindx, rndptindx] + diag(exp(mean_f2_post[rndptindx])/N, 3))

mean_f1_post =  t(cov_matrix_f1[rndptindx, 1:gridsize, drop = FALSE]) %*% postvar_f1 %*% matrix(hstry_mean_rndpt)
cov_matrix_f1_post = cov_matrix_f1 - t(cov_matrix_f1[rndptindx, 1:gridsize, drop = FALSE]) %*%
  postvar_f1 %*% (cov_matrix_f1[rndptindx, 1:gridsize, drop = FALSE])

post_f1_plot_rndpt <- ggplot(data = data.frame(x = c(-4, 4), y = c(-6, 6)), aes(x, y)) +
  geom_function(fun = cos, color = "red") + 
  geom_point(data = data.frame(x = rep(x_grid[rndptindx], each = N), y = c(ystarrnd)), col = "green") +
  geom_line(data = data.frame(x = x_grid, y = mean_f1_post - 1.96 * sqrt(diag(cov_matrix_f1_post))), col = "blue") +
  geom_line(data = data.frame(x = x_grid, y = mean_f1_post + 1.96 * sqrt(diag(cov_matrix_f1_post))), col = "blue") +
  geom_line(data = data.frame(x = x_grid, y = mean_f1_post), col = "black") + ylab("f1(x)")

post_f1_plot_rndpt

#optimization
set.seed(1234)
iter = 1; x_optim = c(); x_optim_pos = c()
log_score = rep(0, length(x_grid)); hstry_mean = hstry_var = c()


cov_f1_chol = t(chol(cov_matrix_f1)); cov_f2_chol = t(chol(cov_matrix_f2));
f1_means = as.vector(cov_f1_chol %*% matrix(rnorm(1 * length(x_grid)), ncol = 1))
f2_means = as.vector(cov_f2_chol %*% matrix(rnorm(1 * length(x_grid)), ncol = 1))
for(i in 1:length(x_grid)){
  
  log_score[i] = log_score[i] + 
    log_score_fun(ystar, mu = f1_means[i], sigmasq = exp(f2_means[i]))
}
#x_optim_pos = append(x_optim_pos, which.min(log_score)); x_optim = x_grid[x_optim_pos]

x_optim_pos = sample.int(gridsize/4, 1); x_optim = x_grid[x_optim_pos]

y_samp = rnorm(N, mean = f1(x_optim), sd = sqrt(exp(f2(x_optim)))); 
hstry = matrix(y_samp, ncol = 1)
hstry_mean = mean(y_samp); hstry_var = var(y_samp)
log_score_hist = log_score
mean_f2_post = rep(0, length(x_grid))

cat(paste("\tItertion", iter, "completed.\n"))

iter_plot <- ggplot(data = data.frame(x = c(-4, 4), y = c(-6, 6)), aes(x, y)) +
  geom_line(data = data.frame(x = x_grid, y = rep(-1.96, N)), col = "blue") +
  geom_line(data = data.frame(x = x_grid, y = rep(1.96, N)), col = "blue") +
  geom_line(data = data.frame(x = x_grid, y = f1_means), col = "black")

iter_plot <- iter_plot + 
  geom_point(data = data.frame(x = rep(x_optim, N), y = c(y_samp)), col = "green")

ppcheck = list()#print(iter_plot)

ppcheck[[1]] = list(iter_plot)

while (iter <= 100) {
  
  postvar_f1 = solve(cov_matrix_f1[x_optim_pos, x_optim_pos] + diag(exp(mean_f2_post[x_optim_pos])/N, iter))
  mean_f1_post =  t(cov_matrix_f1[x_optim_pos, 1:gridsize, drop = FALSE]) %*% postvar_f1 %*% matrix(hstry_mean)
  cov_matrix_f1_post = cov_matrix_f1 - t(cov_matrix_f1[x_optim_pos, 1:gridsize, drop = FALSE]) %*%
    postvar_f1 %*% (cov_matrix_f1[x_optim_pos, 1:gridsize, drop = FALSE])
  
  postvar_f2 = solve(cov_matrix_f2[x_optim_pos, x_optim_pos] + diag(var_f2_error[x_optim_pos], iter))
  mean_f2_post =  t(cov_matrix_f2[x_optim_pos, 1:gridsize, drop = FALSE]) %*% postvar_f2 %*% matrix(log(hstry_var) - mean_f2_error[x_optim_pos])
  cov_matrix_f2_post = cov_matrix_f2 - t(cov_matrix_f1[x_optim_pos, 1:gridsize, drop = FALSE]) %*%
    postvar_f2 %*% (cov_matrix_f2[x_optim_pos, 1:gridsize, drop = FALSE])
  
  cov_f1_chol = t(chol(cov_matrix_f1_post)); cov_f2_chol = t(chol(cov_matrix_f2_post));
  f1_means = as.vector(mean_f1_post) + sqrt(diag(cov_matrix_f1_post)) * rnorm(N) #cov_f1_chol %*% matrix(rnorm(1 * length(x_grid)), ncol = 1))
  f2_means = as.vector(mean_f2_post) + sqrt(diag(cov_matrix_f2_post)) * rnorm(N) #as.vector(mean_f2_post + cov_f2_chol %*% matrix(rnorm(1 * length(x_grid)), ncol = 1))
  remove(log_score); log_score = rep(0, length(x_grid));
  for(i in 1:length(x_grid)){
    
    log_score[i] = log_score[i] + 
      log_score_fun(ystar, mu = f1_means[i], sigmasq = exp(f2_means[i]))
  }
  
  newpos = which.min(log_score); x_optim = x_grid[newpos]
  
  post_f1_plot <- ggplot(data = data.frame(x = c(-4, 4), y = c(-6, 6)), aes(x, y)) +
    geom_function(fun = cos, color = "red") + 
    geom_point(data = data.frame(x = x_grid[x_optim_pos], y = hstry_mean), col = "green") +
    geom_line(data = data.frame(x = x_grid, y = mean_f1_post - 1.96 * sqrt(diag(cov_matrix_f1_post))), col = "blue") +
    geom_line(data = data.frame(x = x_grid, y = mean_f1_post + 1.96 * sqrt(diag(cov_matrix_f1_post))), col = "blue") +
    geom_line(data = data.frame(x = x_grid, y = mean_f1_post), col = "black") + ylab("f1(x)") + 
    geom_vline(xintercept = x_optim, col = "black", linetype = "dotted") +
    geom_vline(xintercept = xstar, col = "red", linetype = "dashed")
  
  post_f2_plot <- ggplot(data = data.frame(x = c(-4, 4), y = c(-6, 6)), aes(x, y)) +
    geom_function(fun = sin, color = "red") + 
    geom_point(data = data.frame(x = x_grid[x_optim_pos], y = log(hstry_var)), col = "green") +
    geom_line(data = data.frame(x = x_grid, y = mean_f2_post - 1.96 * sqrt(diag(cov_matrix_f2_post))), col = "blue") +
    geom_line(data = data.frame(x = x_grid, y = mean_f2_post + 1.96 * sqrt(diag(cov_matrix_f2_post))), col = "blue") +
    geom_line(data = data.frame(x = x_grid, y = mean_f2_post), col = "black") + ylab("f2(x)") +
    geom_vline(xintercept = x_optim, col = "black", linetype = "dotted") +
    geom_vline(xintercept = xstar, col = "red", linetype = "dashed")
  
  #print(gridExtra::grid.arrange(post_f1_plot, post_f2_plot, ncol = 2))
  ppcheck[[iter  + 1]] = list(post_f1_plot, post_f2_plot)  
  log_score_hist = append(log_score_hist, log_score)
  x_optim_pos = append(x_optim_pos, newpos);
  
  
  y_samp = rnorm(N, mean = f1(x_optim), sd = sqrt(exp(f2(x_optim))))
  
  hstry = cbind(hstry, y_samp)
  
  hstry_mean = append(hstry_mean, mean(y_samp)); 
  hstry_var = append(hstry_var, var(y_samp))
  
  iter  = iter + 1
  cat(paste("\tItertion", iter, "completed.\n"))
}

pdf(paste0("bo_onevar_detst_N_", N, ".pdf"), width = 8, height = 4)
gridExtra::grid.arrange(ppcheck[[iter-1]][[1]], ppcheck[[iter-1]][[2]], ncol = 2)
dev.off()

pdf(paste0("logScorePlot_N_", N, ".pdf"), width = 8, height = 4)
ggplot(data = data.frame(x = c(-4, 4), y = c(500, 5000)), aes(x, y)) +
  geom_function(fun = kldiv, args = list(xstar = xstar, ystar = ystar), color = "red") +
  geom_point(data = data.frame(x = x_grid, y = log_score), col = "green") +
  geom_hline(yintercept = min(log_score), col = "blue") +
  geom_vline(xintercept = x_optim, col = "black") + 
  ylab("")
dev.off()