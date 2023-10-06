## ----setup-----------------------------------------------------------------
library("splatter")
library("scater")
library("ggplot2")
set.seed(2021)

# five groups
## ----nGenes----------------------------------------------------------------
# Set the number of genes to 6000
params = newSplatParams()
cells = 4000
genes = 6000
params = setParams(params, list(batchCells = cells,
                                nGenes = genes,
                                group.prob = c(0.05, 0.15, 0.2, 0.25, 0.35),
                                de.prob = c(0.3, 0.1, 0.2, 0.01, 0.1),
                                de.facLoc = c(0.3, 0.1, 0.1, 0.01, 0.2),
                                de.facScale = c(0.1, 0.4, 0.2, 0.5, 0.4))
)
params

# Generate the simulation data using Splatter package,	75%
sim = splatSimulateGroups(params,
                          dropout.shape = c(-0.6,-0.6,-0.6,-0.6,-0.6),
                          dropout.mid = c(2.5,2.5,2.4,2.5,2.4),
                          dropout.type = "group", #确定要模拟的dropout效果的类型
)
#dropout.mid参数控制概率等于0.5的点
#only change dropout.mid to change the sparsity
#80%  dropout.mid = c(3.1,3.1,3.1,3.2,3.1)
#85%  dropout.mid = c(3.8,3.8,3.9,3.9,3.8)
#90%  dropout.mid = c(4.8,4.8,4.8,4.8,4.8)
#95%  dropout.mid = c(6.2,6.2,6.2,6.2,6.2)


sim <- logNormCounts(sim)
sim <- runPCA(sim)
plotPCA(sim, colour_by = "Group")
sim <- runTSNE(sim,ncomponents = 2)
plotTSNE(sim, colour_by = "Group")

simtrue <- as.matrix(sim@assays@data@listData[["TrueCounts"]])
1-sum(simtrue>0)/cells/genes
#simtrue2 <- log10(simtrue+1)

X <- as.matrix(assays(sim)$count)
1-sum(X>0)/cells/genes

real = c(simtrue)
pred = c(X)
obser = c(X)

Omega1 <- (real > 0)
Omega2 <- (obser  == 0)
Omega <- Omega1&Omega2

mae <- mean(abs(real[Omega] -pred[Omega]))
mse <- mean((real[Omega] -pred[Omega])^2)
rmse <- sqrt(mse)
nmse <- mse /mean((real[Omega] -mean(real[Omega]))^2)
cor_pearson_xy <- cor(real,pred, method = 'pearson')

cat("MAE:",round(mae,4),"MSE:",round(mse,4),"RMSE:",round(rmse,4)
    ,"NMSE:",round(nmse,4),"PCC:",round(cor_pearson_xy,4))

simlabel<-sim$Group
label =sub("Group1",1,simlabel,ignore.case =FALSE,fixed=FALSE)
label =sub("Group2",2, label, ignore.case =FALSE, fixed=FALSE)
label =sub("Group3",3, label, ignore.case =FALSE, fixed=FALSE)
label =sub("Group4",4, label, ignore.case =FALSE, fixed=FALSE)
label =sub("Group5",5, label, ignore.case =FALSE, fixed=FALSE)
#label =sub("Group6",6, label, ignore.case =FALSE, fixed=FALSE)

#75  80  85  90  95
write.csv(X,file = "4000sim0.75.csv")
write.csv(simtrue,file = "datasets/4000sim0.75true.csv")
write.csv(label,file = "datasets/4000sim0.75_label.csv")

# write.csv(X,file = "datasets/4000sim0.80.csv")
# write.csv(simtrue,file = "datasets/4000sim0.80true.csv")
# write.csv(label,file = "datasets/4000sim0.80_label.csv")
# 
# write.csv(X,file = "datasets/4000sim0.85.csv")
# write.csv(simtrue,file = "datasets/4000sim0.85true.csv")
# write.csv(label,file = "datasets/4000sim0.85_label.csv")
# 
# write.csv(X,file = "datasets/4000sim0.90.csv")
# write.csv(simtrue,file = "datasets/4000sim0.90true.csv")
# write.csv(label,file = "datasets/4000sim0.90_label.csv")
# 
# write.csv(X,file = "datasets/4000sim0.95.csv")
# write.csv(simtrue,file = "datasets/4000sim0.95true.csv")
# write.csv(label,file = "datasets/4000sim0.95_label.csv")
