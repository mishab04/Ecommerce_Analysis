setwd('C:/Users/Abhinav/Downloads/My Assigments/Customer Analytics/Ecommerce_analysis/Ecommerce_analysis')

ecomdata <- read.csv('data.csv')

summary(ecomdata)

#converting Invoice date to Date format

ecomdata$InvoiceDate <- as.POSIXct(ecomdata$InvoiceDate,format = "%m/%d/%Y %H:%M")

ecomdata$CustomerID <- as.factor(ecomdata$CustomerID)
library(lubridate)

ecomdata$Month <- month(ecomdata$InvoiceDate)

ecomdata$Day <- day(ecomdata$InvoiceDate)

ecomdata$Hour <- hour(ecomdata$InvoiceDate)

ecomdata$Year <- year(ecomdata$InvoiceDate)

ecomdata$Month <- factor(ecomdata$Month,levels = seq(1,12),labels = c('Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'))


library(dplyr)

#Check data for Returns

negquantdata <- ecomdata %>% filter(Quantity< 0)

#negquantdata <- ecomdata[ecomdata$Quantity< 0,]

ecomdata <- ecomdata %>% mutate(Total = Quantity * UnitPrice)

#Summarize the sales by Country, Year, Month

Conttotalsales <- ecomdata %>% group_by(Country) %>% summarise(sales = sum(Total)) %>% arrange(desc(sales))

Contyearsales <- ecomdata %>% group_by(Country,Year) %>% summarise(sales = sum(Total))

Contyearmonthsales <- ecomdata %>% group_by(Country,Year,Month) %>% summarise(sales = sum(Total))

monthsales <- ecomdata %>% group_by(Month) %>% summarise(sales = sum(Total)) %>% arrange(desc(sales))

yearmonthsales <- ecomdata %>% group_by(Year,Month) %>% summarise(sales = sum(Total))

daysales <- ecomdata %>% group_by(Day) %>% summarise(sales = sum(Total)) %>% arrange(desc(sales))

hoursales <- ecomdata %>% group_by(Hour) %>% summarise(sales = sum(Total))

#Creating visualization 

library(ggplot2)

# Removing returns, negtive price & missing customer ID

segdata <- ecomdata %>% filter(Quantity >= 0,UnitPrice >= 0,!is.na(CustomerID))

#Prepraing data for Customer segmentation

custdata <- segdata %>% select(CustomerID,Description)

library(reshape2)

custdata$value <- 1

cust_data <- dcast(custdata,CustomerID ~ Description)

clusterdata <- cust_data[,-1]

#Using PCA to reduce Dimension

prin_comp <- prcomp(clusterdata, scale. = T)

names(prin_comp)

#outputs the mean of variables
prin_comp$center

#outputs the standard deviation of variables
prin_comp$scale

dim(prin_comp$x)

biplot(prin_comp, scale = 0)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
pr_var[1:10]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)

prop_varex[1:20]

#scree plot

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#cumulative scree plot

plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#Based on plot consider only first 1000 PC

pcadata <- data.frame(CustomerID = cust_data$CustomerID,prin_comp$x[,1:1000])

# scaling ( ordering is same)

#pcadata.sc <- scale(pcadata[,-1],center=TRUE,scale=TRUE)

#Elbow Method for finding the optimal number of clusters
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- pcadata
wss <- sapply(10:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(10:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


# optimal number of cluster

library(NbClust)
set.seed(1234)
noculs <- NbClust(pcadata,min.nc=10,max.nc=15,method="kmeans")
table(noculs$Best.n[1,])
barplot(table(noculs$Best.n[1,]), 
        xlab="Numer of Clusters", ylab="Number of Criteria",
        main="Number of Clusters Chosen by 26 Criteria")

library("factoextra")
fviz_nbclust(noculs)

# Elbow method
fviz_nbclust(pcadata, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")
# Silhouette method
fviz_nbclust(pcadata, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
# Gap statistic
# nboot = 50 to keep the function speedy. 
# recommended value: nboot= 500 for your analysis.
# Use verbose = FALSE to hide computing progression.
set.seed(123)
fviz_nbclust(pcadata, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")

#Based on optimal 
set.seed(1234)
results <- kmeans(pcadata, 10)

# understanding the output
results$size #size of each cluster

strength <- results$betweenss/results$tot.withinss # betweenss=distance between cluster should be large
# tot.withinss=avg. distance within cluster should be small
# strength should be higher

library(klaR)

cluster.results <-kmodes(clusterdata,5 )

cluster.results$size

cust_data$Cluster <- cluster.results$cluster

#Let do further analysis for cluster 1

cluster1 <- cust_data %>% filter(Cluster == 1)

clust_data1 <- cluster1[,-1]

cluster1.results <- kmodes(cluster1[,-1],5,iter.max = 10)

cluster1.results$size

# Using K prototype for customer segmentation

library(clustMixType)

kclust <- kproto(clusterdata, k=5, lambda = .8, iter.max = 100, nstart = 1)

clprofiles(kclust, clusterdata)
names(kclust)
str(kclust)

#to check the size of the respective clusters
kclust$size
kclust$cluster


#Prepraing data for market basket analysis

# Removing returns, negtive price & missing customer ID

segdata <- ecomdata %>% filter(Quantity >= 0,UnitPrice >= 0,!is.na(CustomerID))

summary(segdata)

proddata <- aggregate(Description ~ InvoiceNo, data = segdata,paste,collapse = ",")

library(splitstackshape)
custdata[,2]

mbadata <- cSplit(proddata, 'Description', sep=",", type.convert=FALSE)

#Conducting Market Basket Analysis

mbadata$InvoiceNo <- NULL

write.csv(mbadata,"mbadata.csv",quote = F,row.names = FALSE)

library(arules)
library(arulesViz)

itemdata <- read.transactions("mbadata.csv",sep = ",")

summary(itemdata)

itemFrequencyPlot(itemdata, topN = 20)

itemFrequencyPlot(itemdata, support = 0.03)

# we can control no of items by specifiying minlen and maxlen

itemrules <- apriori(itemdata, parameter = list(support = 0.001, confidence = 0.9, maxlen = 3))

#inspecting top 10 rules

inspect(itemrules[1:10])

#Removing duplicate rules

itemrules <- itemrules[!is.redundant(itemrules)]


itemrules<-sort(itemrules, by="confidence", decreasing=TRUE)


itemrules_df <- as(itemrules, "data.frame")

#Creating a visualization of the rules

topRules <- itemrules[1:10]

plot(topRules)

plot(topRules, method="graph")

plot(toprules,method="graph",interactive=TRUE,shading=NA)

plot(topRules, method = "grouped")





