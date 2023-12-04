
################### ETAPE 1 #######################
# Charger les données
data <- read.csv("C:/Users/LF ELEC/Desktop/CNAM/STA- 211/ionosphere.csv",
                 col.names=c("cible","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10",
                             "x11","x12","x13","x14","x15","x16","x17","x18","x19","x20",
                             "x21","x22","x23","x24","x25","x26","x27","x28","x29",
                             "x30","x31","x32","x33","x34"))
data$cible=as.factor(data$cible)
View(data)

################## ETAPE 2-1 #######################
# Faire les statistiques descriptives selon la variables cible
###Methode 1: en utilisant la fonction "summary"
str(data)
summary(data)
###Methode 1: en utilisant la fonction "by"
by(data[,2:35], data$cible, summary)
### Mehode 2: en utilisant la fonction "describe"
library(Hmisc)
describe(data)
### Methode 3: en utilisant la fonction "describeBy"
library(psych)
describeBy(data,data$cible)

#################### ETAPE 2-2 #############################
# Representation graphique des variables



################ ETAPE 3 ######################
# Trouver la relation entre les variables
## Verifier si la base donnee est lineaire ou non-lineaire
## Eliminer la variable cible "class"
## Eliminer la variable x1 car elle contienne des valeurs zero=constante
## Eliminer la variable x34 derniere variable car elle est qualitative
d=data[,c(3:33)]

######## METHODE 1: Analyse factorielle "ACP"
#1) Effectuer l'ACP sur une base de donnee normalisee
pca <- prcomp(d, scale = TRUE)
#2) Visualiser les résultats
### le biplot nous donne un graphe a 2 dimension pour visualizer en meme temps
##-->la relation entre les observations: sous forme de points
##-->la relation entre les variables: sous forme de flèches
biplot(pca, cex = 0.8)
#3) Standardiser les données
data_std <- scale(d)
#4) Effectuer la PCA
pca <- prcomp(data_std)
#5) Obtenir les proportions de variance expliquée par chaque composante principale
prop_var <- pca$sdev^2 / sum(pca$sdev^2)
#6) Afficher le graphique de la variance expliquée
plot(prop_var, type="b", xlab="Composante principale", ylab="Proportion de variance expliquée")


########### METHODE 2: Traouver la correlation entre les variables
library(psych)
#pour les variables: 
pairs.panels(d[,1:5], 
             method = "pearson", # correlation method
             hist.col = '#E5FF00',
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses#  
             ) 

pairs.panels(d[,6:11], 
             method = "pearson", # correlation method
             hist.col = '#E5FF00',
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses#  
             ) 

pairs.panels(d[,12:17], 
             method = "pearson", # correlation method
             hist.col = '#E5FF00',
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses#  
             ) 

pairs.panels(d[,18:22], 
             method = "pearson", # correlation method
             hist.col = '#E5FF00',
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses#  
             ) 

pairs.panels(d[,23:27], 
             method = "pearson", # correlation method
             hist.col = '#E5FF00',
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses#  
             ) 

pairs.panels(d[,28:31], 
             method = "pearson", # correlation method
             hist.col = '#E5FF00',
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses# 
             ) 


################## ETAPE 4 #######################
#### Creer une partition
library(caret)
set.seed(14)
data=as.data.frame(data)
pard1 <- createDataPartition(data$cible,p=0.8,list=F) 
train <- data[pard1,]
test <- data[-pard1,]
nrow(test)
#Eliminant la variable "x1" puisque c'est une constante
train <- train[, -which(names(train) == "x1" )]
test <- test[,-which(names(test) == "x1")]


################## ETAPE 5: CAS 1:selon "Tune" #######################
### Appliquer le modele "SVM" 
############## CAS 1: CHOISIR LES PARAMETRES SELON "TUNE"
### pour trouver les meilleurs parametres en utilisant la fct "tune"
### pour appeler la fct svm en utilse : "e1071::svm"
library(e1071)
svm_tune=tune(e1071::svm, cible~.,data=train, 
              ranges=list(cost=10^(-1:2), 
          gamma=c(.5,1,2),kernel=c("radial","polynomial","sigmoid","linear")))
print(svm_tune)
#### on obtient les parametres optimale choisi par tune
#################### modele finale #################
svm_f <- svm(cible ~ ., data=train, kernel="polynomial", cost=0.1, gamma=0.5)
### on voit que le nbr du support vector machine a augmenter
summary(svm_f)
## pour voir tous les nom du variables
print(attributes(svm_f))
head(train,23)
# numero d'ind points support
print((rownames(train))[svm_f$index])

# prediction
pred_1 <- predict(svm_f,test)
pred_1
#Methode 1: pour evaluer la performance: "Matrice de confusion"
t1=table(pred_1,test$cible) ; t1 ;
m1=confusionMatrix(pred_1,test$cible) ; m1 ;
#Methode 2: pur evaluer la performance: "Taux d'erreur"
e1=1-sum(diag(t1))/sum(t1) ; e1 ;
#Methode 3: Pour evaluer la performance; "Courbe de Roc"
# Prédire les valeurs pour les données de test
svm_scores_1 <- predict(svm_f, newdata = test, type="decision")
#Convertir le svm_score en numeric pas en facteur
svm_scores_1=as.numeric(svm_scores_1)
# Calculer les performances
library(pROC)
roc_obj_1 <- roc(test$cible, svm_scores_1)
auc_1 <- auc(roc_obj_1)
# Tracer la courbe ROC
plot(roc_obj_1, main = paste("Courbe ROC SVM non linéaire avec AUC =", round(auc_1, 2)), col = "blue")
# pour trouver l'intervale de confiance en utilise fct "ci" 
#La fonction ci calcule l'intervalle de confiance de l'AUC de la courbe ROC et 
#retourne un objet de type ci.info, contenant les bornes inférieure et supérieure
#de l'intervalle de confiance ainsi que le niveau de confiance.
ci_obj_1 <- ci(roc_obj_1)
ci_obj_1
# pour trouver le seuil optimal de CI on utilise fcy "coords"
#La fonction coords calcule les coordonnées (sensibilité, spécificité) 
#correspondant à différents seuils de classification et retourne un objet de type coords.info,
#contenant une matrice avec deux colonnes correspondant aux coordonnées (sensibilité, spécificité)
#et un vecteur avec les seuils de classification correspondants.
coords_obj_1 <- coords(roc_obj_1, "best", ret = c("threshold", "specificity", "sensitivity"))
coords_obj_1


#############################################################
################## ETAPE 5: CAS 2:choisir aleatoirement" ##################
##### ce qu'on a changer c'est la noyau "kernel" 
##on a considere que notre base de donnee est lineaire ce qui n'est pas vrai
##Mais on a obtenu un resultats plus efficace que le 1er cas
library(e1071)
svm_2 <- svm(cible ~ ., data=train, kernel="linear", cost=0.1, gamma=0.5)
### on voit que le nbr du support vector machine a augmenter
summary(svm_2)
## pour voir tous les nom du variables
print(attributes(svm_2))
head(train,23)
# numero d'ind points support
print((rownames(train))[svm_2$index])

# prediction
pred_2 <- predict(svm_2,test)
pred_2
#Methode 1: pour evaluer la performance: "Matrice de confusion"
t2=table(pred_2,test$cible) ; t2 ;
m2=confusionMatrix(pred_2,test$cible) ; m2 ;
#Methode 2: pur evaluer la performance: "Taux d'erreur"
e2=1-sum(diag(t2))/sum(t2) ; e2 ;
#Methode 3: Pour evaluer la performance; "Courbe de Roc"
# Prédire les valeurs pour les données de test
svm_scores_2 <- predict(svm_2, newdata = test, type="decision")
#Convertir le svm_score en numeric pas en facteur
svm_scores_2=as.numeric(svm_scores_2)
# Calculer les performances
library(pROC)
roc_obj_2 <- roc(test$cible, svm_scores_2)
auc_2 <- auc(roc_obj_2)
# Tracer la courbe ROC
plot(roc_obj_2, main = paste("Courbe ROC SVM non linéaire avec AUC =", round(auc_2, 2)), col = "blue")
# pour trouver l'intervale de confiance en utilise fct "ci" 
#La fonction ci calcule l'intervalle de confiance de l'AUC de la courbe ROC et 
#retourne un objet de type ci.info, contenant les bornes inférieure et supérieure
#de l'intervalle de confiance ainsi que le niveau de confiance.
ci_obj_2 <- ci(roc_obj_2)
ci_obj_2
# pour trouver le seuil optimal de CI on utilise fcy "coords"
#La fonction coords calcule les coordonnées (sensibilité, spécificité) 
#correspondant à différents seuils de classification et retourne un objet de type coords.info,
#contenant une matrice avec deux colonnes correspondant aux coordonnées (sensibilité, spécificité)
#et un vecteur avec les seuils de classification correspondants.
coords_obj_2 <- coords(roc_obj_2, "best", ret = c("threshold", "specificity", "sensitivity"))
coords_obj_2


###########################################################################
################### ETAPE 5: CAS 3: En utilisant librar "Rminer" ############
############ CAS 3: CHOISIR LES PARAMETRES SELON "RMINER"
## c'est une boite noir on ne choisit pas les parametres 
#### Creer une partition
library(caret)
set.seed(14)
data=as.data.frame(data)
pard1 <- createDataPartition(data$cible,p=0.8,list=F) 
train <- data[pard1,]
test <- data[-pard1,]
#Eliminant la variable "x1" puisque c'est une constante
train <- train[, -which(names(train) == "x1" )]
test <- test[,-which(names(test) == "x1")]
# En utilisant la librairy "rminer"
library(rminer)
train_clean <- na.omit(train)
test_clean <- na.omit(test)
# Creer le modele en utilisant "rminer"
SVM_model_rminer <- fit(cible ~ ., data=train_clean, model="svm", kernel="rbfdot", kpar=list(sigma=0.1))
# Faire les prédictions
predictions_3 <- predict(SVM_model_rminer, newdata = test_clean, type="class")
# Methode 1: Pour evaluer la performance : "Taux d'erreur" 
error_rate_3 <- sum(predictions_3 != test_clean$cible) / nrow(test_clean)
# Afficher l'erreur
cat("Taux d'erreur: ", error_rate_3, "\n")
#Methode 2: Pour evaluer la performance; "Courbe de Roc"
# Créer un objet "roc" en utilisant la fonction "roc" de la bibliothèque "pROC"
library(pROC)
roc_obj_3 <- roc(test_clean$cible, predictions_3[,2])
auc_roc_3 <- auc(roc_obj_3)
# Tracer la courbe ROC
plot(roc_obj_3, main = paste("Courbe ROC SVM non linéaire avec AUC =",
       round(auc_roc_3, 2)), col = "blue")
# Afficher l'aire sous la courbe (AUC)
cat("Aire sous la courbe ROC: ", auc(roc_obj_3), "\n")
# pour trouver l'intervale de confiance en utilise fct "ci" 
#La fonction ci calcule l'intervalle de confiance de l'AUC de la courbe ROC et 
#retourne un objet de type ci.info, contenant les bornes inférieure et supérieure
#de l'intervalle de confiance ainsi que le niveau de confiance.
ci_obj_3 <- ci(roc_obj_3)
ci_obj_3
# pour trouver le seuil optimal de CI on utilise fcy "coords"
#La fonction coords calcule les coordonnées (sensibilité, spécificité) 
#correspondant à différents seuils de classification et retourne un objet de type coords.info,
#contenant une matrice avec deux colonnes correspondant aux coordonnées (sensibilité, spécificité)
#et un vecteur avec les seuils de classification correspondants.
coords_obj_3 <- coords(roc_obj_3, "best", ret = c("threshold", "specificity", "sensitivity"))
coords_obj_3



### R.Q : Dans ce cas on ne peut pas trouver la matrice de confusion car 
##les dimensions des prédictions et des données de test ne sont pas égales.
##Cela peut arriver lorsque la variable cible est déséquilibrée dans les donnée
##s de test, ce qui peut affecter les performances de la matrice de confusion

###################### FINALEMENT ########################
## Montrant les 3 courbes de roc dans une seul figure:
#Multiroc
library(pROC)
rocad=plot.roc(test$cible, svm_scores_1, percent=TRUE, col="blue",
               lwd=2, print.auc=TRUE, print.auc.y=40)
rocad=plot.roc(test$cible, svm_scores_2, percent=TRUE, col="green",
               lwd=2, print.auc=TRUE, add=TRUE, print.auc.y=30)
rocad=plot.roc(test_clean$cible, predictions_3[,2], percent=TRUE, col="red",
               lwd=2, print.auc=TRUE, add=TRUE, print.auc.y=20)
#Add legend
legend("bottomright",legend=c("SVM avec Tune","SVM aleatoire","SVM avec rminer"),
       col=c("blue","green","red"),
             lwd=5, cex=0.6, xpd=TRUE, horiz=TRUE)


###################### FINALEMENT ########################
library(microbenchmark)
library("caret")
timbag= microbenchmark(times=10,unit = "ms", 
                       
                       "SVM avec Tune"={svm_tune=tune(e1071::svm, cible~.,data=train, 
                                    ranges=list(cost=10^(-1:2), 
                                 gamma=c(.5,1,2),kernel=c("radial","polynomial","sigmoid","linear")))
                       
                       } ,
                       
                       "SVM avec parametre aleatoire"={ svm_2 <- svm(cible ~ ., data=train,
                                                  kernel="linear", cost=0.1, gamma=0.5)
                       },   
                       
                       "SVM avec Rminer"={SVM_model_rminer <- fit(cible ~ ., data=train_clean, model="svm", kernel="rbfdot", kpar=list(sigma=0.1))

                       }   
                       
)

#pour comparer graphiquement le temps de chaque cas de bagging-->fonction "autoplot"
#pour trouver la moyenne de temps de chaque cas de bagging-->fonction "summary"
### c'est le temps d'excusion de chaque methode
autoplot(timbag)
summary(timbag)
