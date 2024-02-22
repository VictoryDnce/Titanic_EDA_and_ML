import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
import matplotlib
matplotlib.use("Qt5Agg")
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("free_work/Titanic/train.csv")
train = df.copy()
test = pd.read_csv("free_work/Titanic/test.csv")


#--------------------------------- Data Preprocessing ----------------------------------------------
"""
plt.style.use("seaborn-v0_8-notebook")
plt.style.use("default")
plt.plot(df["Age"])
"""

train.head()
train.shape
train.info()
train.isnull().sum()
train.describe().T

# ---------------------------------- EDA(Exploratory Data Analysis) -----------------------------------

# ------------------------ Handling Missing Values ------------------------------------------
length =  len(train)
train = pd.concat([train,test],axis=0,ignore_index=True)
train.isnull().sum()

##############
# Embarked
##############

train.loc[train["Embarked"].isnull()]

# train.boxplot(column="Fare",by="Embarked")
# plt.show()

# train[["Fare","Embarked"]].groupby("Embarked",as_index = False).mean()

plt.figure(figsize=(8,6))
ax = sns.barplot(data=train, x="Embarked",y="Fare",palette="rocket_r")
for i in ax.containers:
    ax.bar_label(i,label_type="center")
ax.set_title("Average Fare by Classes")
plt.xlabel("Class")
plt.ylabel("Average Fare")
plt.tight_layout()
plt.show()

# it seems that class C has the highest average fare among other classes, so we can fill in the NaN values with C.

train["Embarked"].fillna("C",inplace=True)

##############
# Fare
##############

train.loc[train["Fare"].isnull()]

train[["Fare","Pclass"]].groupby("Pclass").mean()

# Since the NaN value belongs to Pclass 3, it can be filled with the average fare.

train["Fare"].fillna(np.mean(train.loc[train["Pclass"]==3,"Fare"]),inplace=True)

##############
# Age
##############

train.groupby("Sex")["Age"].mean()
sns.catplot(x = "Sex", y = "Age", data = train, kind = "box")
plt.show()

# The sex variable does not help us fill in the NaN values because the average is almost the same.

# --------------------------------------------------------------------------------------------
fig, ax = plt.subplots(3,2,figsize=(16,9))

# Pclass --------------------------
# plt.figure(figsize=(8,6))
sns.barplot(data=train, x="Pclass",y="Age",palette="rocket_r",estimator="median",ax=ax[0,0])
for i in ax[0,0].containers:
    ax[0,0].bar_label(i,label_type="center",fmt='%.0f')
ax[0,0].set_title("Median Age According to Pclass")

# plt.figure(figsize=(8, 6))
sns.barplot(data=train, x="Pclass", y="Age", palette="viridis", estimator="mean",ax=ax[0,1])
for i in ax[0,1].containers:
    ax[0,1].bar_label(i, label_type="center", fmt='%.0f')
ax[0,1].set_title("Mean Age by Pclass")

# sns.catplot(x = "Pclass", y = "Age", data = train, kind = "box")
# plt.show()



# SibSp --------------------------
# plt.figure(figsize=(8,6))
sns.barplot(data=train, x="SibSp",y="Age",palette="rocket_r",estimator="median",ax=ax[1,0])
for i in ax[1,0].containers:
    ax[1,0].bar_label(i,label_type="center",fmt='%.0f')
ax[1,0].set_title("Median Age According to SibSp")

# plt.figure(figsize=(8, 6))
sns.barplot(data=train, x="SibSp", y="Age", palette="viridis", estimator="mean",ax=ax[1,1])
for i in ax[1,1].containers:
    ax[1,1].bar_label(i, label_type="center", fmt='%.0f')
ax[1,1].set_title("Mean Age by SibSp")

# sns.catplot(x = "SibSp", y = "Age", data = train, kind = "box")
# plt.show()



# Parch --------------------------
# plt.figure(figsize=(8,6))
sns.barplot(data=train, x="Parch",y="Age",palette="rocket_r",estimator="median",ax=ax[2,0])
for i in ax[2,0].containers:
    ax[2,0].bar_label(i,label_type="center",fmt='%.0f')
ax[2,0].set_title("Median Age According to Parch")

# plt.figure(figsize=(8, 6))
sns.barplot(data=train, x="Parch", y="Age", palette="viridis", estimator="mean",ax=ax[2,1])
for i in ax[2,1].containers:
    ax[2,1].bar_label(i, label_type="center", fmt='%.0f')
ax[2,1].set_title("Mean Age by Parch")
plt.tight_layout()

# sns.catplot(x = "Parch", y = "Age", data = train, kind = "box")
# plt.show()
# --------------------------------------------------------------------------------------------

# When we consider age, according to the graphs, there are 2 criteria for Parch, SibSp, Pclass, which are median and mean. I choose mean to fill NaN values

# --- Filling

indexes = list(train["Age"][train["Age"].isnull()].index)
for i in indexes:
    mean_age = train["Age"][((train["Pclass"] == train.iloc[i]["Pclass"])&\
                             (train["SibSp"] == train.iloc[i]["SibSp"])&\
                             (train["Parch"] == train.iloc[i]["Parch"]))].mean()

    # If a NaN value comes up, I will fill it with the general age mean.
    mean_age_general = train["Age"].mean()

    if not np.isnan(mean_age):
        train["Age"].iloc[i] = round(mean_age)
    else:
        train["Age"].iloc[i] = round(mean_age_general)


# ---------------------------------- Feature Engineering -----------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# Age ---------------------

train.loc[train["Age"]<35,"Age_Category"] = "Young"
train.loc[(35<=train["Age"]) & (train["Age"]<=55),"Age_Category"] = "Middle_Age"
train.loc[55<train["Age"],"Age_Category"] = "Old"

# Changing the order of columns
col = train.pop('Age_Category')
train.insert(6, col.name, col)

train.head()


 # Name ---------------------
train["Name"][1].split(".")[0].split(",")[1].strip()

train["Title"] = [val.split(".")[0].split(",")[1].strip() for val in train["Name"]]

# Changing the order of columns
col = train.pop('Title')
train.insert(4, col.name, col)

train.head()

sns.countplot(data=train,x="Title")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


train["Title"] = [val if val=="Mr" or val=="Mrs" or val=="Miss" or val=="Master" or val=="Mlle" else "Other"  for val in train["Title"]]

# Mlle: matmazel(Miss)
train["Title"].replace("Mlle","Miss",inplace=True)

sns.countplot(data=train,x="Title")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Family Size ---------------------

# Family_Size = Siblings/Spouses + Parents/Children + itself
train["Family_Size"] = train["SibSp"] + train["Parch"] + 1
train["Family_Size_Category"] = ["Small_F" if val<5 else "Big_F" for val in train["Family_Size"]]

# Changing the order of columns -- Family_Size
col = train.pop('Family_Size')
train.insert(10, col.name, col)

# Changing the order of columns -- Family_Size_Category
col = train.pop('Family_Size_Category')
train.insert(11, col.name, col)

train.head()


# ---------------------------------- Data Analysis & Visualization -----------------------------------
plt.style.use("default")
# pd.crosstab(index=df["Survived"],columns=df["Pclass"])

# Sex-Survived ---------------------
rate = train[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean()

sns.set_theme(palette="viridis", font="arial", font_scale=1.5)

fig2, axs = plt.subplots(figsize=(14, 8))

plt.pie(data=train,x=rate["Survived"], labels=rate["Sex"].unique(), autopct='%.1f%%',shadow=True,explode=[0.05,0.05], startangle = 90,wedgeprops= {"edgecolor":"black",'linewidth': 1,'antialiased': True})
plt.title("Survival Rate by Sex")
plt.show()

# Among passengers, women have a much better survival rate than men


plt.style.use("default")
plt.style.use("seaborn-v0_8-darkgrid")
plt.style.use("dark_background")

# Siblings/Spouses-Survived ---------------------
sns.catplot(x = "SibSp", y = "Survived", data = train, kind = "bar",errwidth=0,palette="YlOrBr")
plt.show()

# Passengers with SibSp numbers between 0 and 2 have a higher chance of survival than others


# Parents/Children-Survived ---------------------
sns.catplot(x="Parch", y ="Survived", data = train, kind="bar",errwidth=0,palette="crest")
plt.show()

# Parch numbers seems important for survival

# Family Size-Survived ---------------------

sns.catplot(x="Family_Size", y ="Survived", data = train, kind="bar",errwidth=0,palette="mako")
plt.show()

sns.catplot(x="Family_Size_Category", y ="Survived", data = train, kind="bar",errwidth=0,palette="cubehelix")
plt.show()

# Large families, especially families larger than 4, seem to have little chance of survival


# Passenger Class-Survived ---------------------
sns.catplot(x="Pclass", y ="Survived", data = train, kind="bar",errwidth=0,palette="flare")
plt.show()

# It is not difficult to guess that the first class passengers survived


# Embarked-Survived ---------------------
g = sns.catplot(x="Embarked", y ="Survived", data = train, kind="bar",errwidth=0,palette="dark:salmon_r")
g.set_xticklabels(["Southampton", "Cherbourg", "Queenstown"])
plt.show()

#it seems that Passengers embarked from Cherbourg have a higher survival rate


# Title-Survived ---------------------
sns.catplot(data=train, x="Title", y="Survived",kind="bar",errwidth=0)
plt.show()

# The survival rate of married women is slightly higher than that of unmarried women, and the survival rate of men is also observed to be lower in this graph

plt.style.use("default")
plt.style.use("seaborn-v0_8-whitegrid")

# Age-Survived ---------------------

sns.FacetGrid(train, col = "Survived").map(sns.distplot, "Age", bins = 25)
plt.show()

# The average age of passengers is between 20 and 40
# Mortality rate is quite high between the ages of 20-30
# Young children have a high survival rate
# According to the graph, there are survivors at the age of 80

# Fare-Survived ---------------------
sns.FacetGrid(train, col = "Survived").map(sns.distplot, "Fare", bins = 5)
plt.show()

# g = sns.FacetGrid(train, row = "Pclass", col = "Survived")
# g.map(plt.hist,  "Fare")
# g.add_legend()
# plt.show()

# We can say that the passengers who paid high fares survived.

"""
# Cinsiyete göre hayatta kalma oranı
train[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean()

# Cinsiyete göre hayatta kalma sayıları
pd.crosstab(index=train["Survived"],columns=train["Sex"]).reset_index()
train.groupby(["Sex","Survived"])["Survived"].count().unstack()
"""


# ---------------------------------- Preparation of Data -----------------------------------

train = pd.get_dummies(train,columns = ["Title","Sex","Age_Category","Family_Size_Category","Embarked"],drop_first=True,dtype=int)

length

test = train[length:]
test_Pass_Id = test[["PassengerId"]]
test.drop(["PassengerId","Survived","Name","Ticket","Cabin"],axis=1,inplace=True)

test.head()

train = train[:length]

train.head()

X = train.drop(["PassengerId","Survived","Name","Ticket","Cabin"],axis=1)
y = train[["Survived"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=0)

"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

rb = RobustScaler()
X_train = rb.fit_transform(X_train)
X_test = rb.transform(X_test)
"""

# ---------------------------------- Modelling with Keras -----------------------------------
"""
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

classifier = Sequential()

classifier.add(Dense(units=16,activation="relu",input_dim=16))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units=8,activation="relu"))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units=1,activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)

classifier.fit(X_train,y_train,epochs=300,verbose=1,validation_data=(X_test,y_test),callbacks=[earlyStopping], validation_split=0.2)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
accuracy_score(y_test,y_pred)
print(classification_report(y_pred,y_test))

# ---------------------------------------------------------------------

modelKaybı = pd.DataFrame(classifier.history.history)
modelKaybı[["loss","val_loss"]].plot()

# 0.8305084745762712
# 100 - 0.8305084745762712
# 300 - 0.8169491525423729
# 150 - 0.8169491525423729
"""
# ---------------------------------- Modelling with XGBoost -----------------------------------
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)

accuracy_score(y_test,y_pred)

print(classification_report(y_pred,y_test))


# ------------------------------------------ Model Optimization----------------------------------
xgb_optimized = XGBClassifier(
 learning_rate = 0.02,
 n_estimators= 1000,
 max_depth= 6,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)

xgb_optimized.fit(X_train,y_train)
y_pred = xgb_optimized.predict(X_test)
accuracy_score(y_test,y_pred) # 0.8576271186440678
print(classification_report(y_pred,y_test))

# 0.8542372881355932
# 0.8542372881355932
# 0.8542372881355932
# 0.8542372881355932
# 0.8576271186440678


# ------------------------------------------ Feature Importance ----------------------------------

xgb_fea_imp = pd.DataFrame({"Value": xgb_optimized.feature_importances_, "Feature": X_train.columns})

fig3, ax = plt.subplots(figsize = (10,7))

sns.barplot(x="Value",y="Feature",data=xgb_fea_imp.sort_values(by="Value",ascending=False)[0:len(X_train)]).set_title("Feature Importance for XGBoost")

plt.tight_layout()

# ------------------------------------------ Submission ----------------------------------

predictions = xgb_optimized.predict(test)

results = pd.DataFrame({"PassengerId": test_Pass_Id["PassengerId"], "Survived": predictions})
results.to_csv("submission.csv",index=False)

results.reset_index(drop=True).head()