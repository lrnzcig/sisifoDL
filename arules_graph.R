# https://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data
# https://archive.ics.uci.edu/ml/machine-learning-databases/anonymous/

msweb.orig <- read.csv("../Downloads/anonymous-msweb.data",
                       header=FALSE,
                       sep=",",
                       col.names=c("V1", "V2", "V3", "V4", "V5", "V6"))

attribute.lines <- msweb.orig[msweb.orig$V1 == "A", c("V2", "V4", "V5")]
colnames(attribute.lines) <- c("id", "title", "url")
head(attribute.lines)

casevote.lines <- msweb.orig[msweb.orig$V1 == "C" | msweb.orig$V1 == "V", c("V1", "V2")]
head(casevote.lines)

casevote.lines$rownames <- as.numeric(rownames(casevote.lines)) # add indexes as a new column
dt <- data.table(casevote.lines[, !(names(casevote.lines)
                                    %in% "parent")]) # convert to dataframe, excluding the parent column from a previos trial
head(dt)

dt[V1 == "C", parent:=rownames, by=rownames] # parents of C lines are themselves
difference <- 1
while (sum(is.na(dt$parent)) > 0) {
  dt[, shiftV1 := shift(V1, difference)] # shift by "difference"
  dt[shiftV1 == "C" & is.na(parent),
     parent:=(rownames-difference), by=rownames] # set parent value of visits with n. of pages == "difference"
  difference <- difference + 1
}
casevote.lines$parent = dt$parent
head(casevote.lines)


clicks <- merge(x=casevote.lines[casevote.lines$V1 == "C", c("V2", "parent")],
                y=casevote.lines[casevote.lines$V1 == "V", c("V2", "parent")],
                by="parent")
colnames(clicks) <- c("remove", "userId", "attributeId")
head(clicks)


msweb <- merge(x=clicks[,c("userId", "attributeId")],
               y=attribute.lines,
               by.x=c("attributeId"),
               by.y=c("id"))
msweb <- msweb[order(msweb$userId),]
head(msweb)


msweb.items <- data.table(msweb[, c("title", "userId")])
msweb.items[, l:=.(list(unique(title))), by=userId] # creates list of pages per user, see note
msweb.items <- msweb.items[! duplicated(userId), l] # removes duplicated lines per user and selects only the list 

head(msweb.items, 3)


# convertir a matriz de adyacencias
msweb.adj <- data.table::dcast(msweb, as.formula("userId~title"),
                               value.var="title", fun.aggregate=function(x) as.integer(length(x) >= 1))
value_names <- names(msweb.adj)[names(msweb.adj) != "userId"] 
value_names <- make.names(value_names)
names(msweb.adj) <- c("userId", value_names)
head(msweb.adj)


library(netCoin)


C <- coin(msweb.adj[, value_names]) # coincidence matrix

N <- asNodes(C) # node data frame
E <- edgeList(C) # edge data frame

Net <- netCoin(N, E) # network object

plot(Net)

