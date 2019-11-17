# https://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data
# https://archive.ics.uci.edu/ml/machine-learning-databases/anonymous/

msweb.orig <- read.csv("~/Downloads/clickstream/anonymous-msweb.data",
                       header=FALSE,
                       sep=",",
                       col.names=c("V1", "V2", "V3", "V4", "V5", "V6"))