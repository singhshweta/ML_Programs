# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:04:49 2017

@author: hp
"""

##Random Forest Clasifier used##
#We tweak the style of this code a little bit to have centered plots.
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

# remove warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

test_data=pd.read_csv('C:\\Users\\inspiron\\Desktop\\titanic_survival\\titanic survival test.csv',encoding = "iso-8859-15")
train_data=pd.read_csv('C:\\Users\\inspiron\\Desktop\\titanic_survival\\titanic survival train.csv',encoding = "iso-8859-15")

print("train data before filling missing values of age:\n",train_data.describe())
#filling null values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

print(train_data)
print("shape of train_data=",train_data.shape)
print("header of train_data=\n",train_data.head())
print("Analysis of train_data:\n",train_data.describe())
                              

#lets analyse survival on basis of sex
survived_sex = train_data[train_data['Survived']==1]['Sex'].value_counts()
dead_sex = train_data[train_data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
print("\nsurvived sex:\n ",survived_sex)
print("\ndead sex: \n",dead_sex)
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8),label = ['Male','Female'])    #red and blue are default colors

#lets analyse survival on basis of age
figure = plt.figure(figsize=(15,8))
plt.hist([train_data[train_data['Survived']==1]['Age'], train_data[train_data['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'], bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
#These two first charts confirm that one old code of conduct that sailors 
#and captains follow in case of threatening situations: "Women and children first !".

#lets analyse survival on basis of fare
figure = plt.figure(figsize=(15,8))
plt.hist([train_data[train_data['Survived']==1]['Fare'],train_data[train_data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()
#passengers with more expensive tickets, and therefore a more important social status, seem to be rescued first.

#Let's now combine the age, the fare and the survival on a single chart.
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(train_data[train_data['Survived']==1]['Age'],train_data[train_data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(train_data[train_data['Survived']==0]['Age'],train_data[train_data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
#A distinct cluster of dead passengers (the red one) appears on the chart. 
#Those people are adults (age between 15 and 50) of lower class (lowest ticket fares).

#In fact, the ticket fare correlates with the class as we see it in the chart below.
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.set_ylabel('Average fare')
train_data.groupby('Pclass').mean()['Fare'].plot(kind='bar', ax = ax)
plt.show()

#Let's now see how the embarkation site affects the survival.
survived_embark = train_data[train_data['Survived']==1]['Embarked'].value_counts()
dead_embark = train_data[train_data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar', stacked=True, figsize=(15,8))
#There seems to be no distinct correlation here.

##Feature engineering##
#we couldn't manage to analyze more complicated features like the names or the tickets
#because these required further processing.  we'll focus on the ways to transform these
#specific features in such a way they become easily fed to machine learning algorithms.
#We'll also create, or "engineer" some other features that'll be useful in building the model.

#let's define a print function that asserts whether or not a feature has been processed.
def status(feature):
    print('Processing',feature,': ok')

#combining train and test sets will save us some repeated work to do later on when testing.
def get_combined_data():
    train = pd.read_csv('C:\\Users\\inspiron\\Desktop\\titanic_survival\\titanic survival train.csv')
    test = pd.read_csv('C:\\Users\\inspiron\\Desktop\\titanic_survival\\titanic survival test.csv')
    train.drop('Survived', 1, inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    return combined

combined = get_combined_data()
print("\nshape of combined data: ",combined.shape)
print(combined.head())

#processing names
#lets extract out titles from passenger's name to find their social status
def get_titles():
    global combined
    # we extract the title from each name
    # in Python anonymous functions(lambda functions) are defined using the lambda keyword.
    #Anonymous functions are those that are defined without a name.
    #normal functions use the keyword def while anonymous functions use lambda.
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)

get_titles()
print("\ncombined data with additional column 'Title':\n",combined.head())

#processing ages
#We have seen in the first part that the Age variable was missing 177 values. This is a large
#number ( ~ 13% of the dataset). Simply replacing them with the mean or the median age might
#not be the best solution since the age may differ by groups and categories of passengers.

#To avoid data leakage from the test set, we perform these operations separately on the train 
#set and the test set. Data leakage in sense that while calculating medians, separate medians 
#should be calculated for train and test data.
grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])
grouped_median_test = grouped_test.median()

print("grouped_median_train:\n",grouped_median_train)
print("grouped_median_test:\n",grouped_median_test)

#Let's create a function that fills in the missing age in combined based on these different attributes.
def process_age():
    global combined
    
    # a function that fills the missing values of the Age variable
    def fillAges(row, grouped_median):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']
    
    combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    status('age')

print(process_age())
print("after filling missing values of age:\n",combined.info())
#However, we notice a missing value in Fare, two missing values in Embarked and a lot of missing values in Cabin. 

#Let's now process the names.
def process_names():   
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)   
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)   
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)   
    status('names')
    
process_names()
print("\ndata with encoded attributes: \n",combined.head())

#processing fare
def process_fares():
#This function simply replaces one missing Fare value by the mean.
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)
    status('fare')

process_fares()

#processing embarked
def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)  
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    status('embarked')

process_embarked()
print("\ndata with encoded attributes: \n",combined.head())

#processing cabin
def process_cabin():
    
    global combined
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined,cabin_dummies], axis=1)
    combined.drop('Cabin', axis=1, inplace=True)
    status('cabin')

process_cabin()

print(combined.info())      #this result shows there are no more missing values

#processing sex
def process_sex():
    
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    status('sex')
    
process_sex()

#processing Pclass
def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    status('pclass')

process_pclass()

#processing ticket
def process_ticket():
#This functions preprocess the tikets first by extracting the ticket prefix. When it fails in extracting a prefix it returns XXX.
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()         #by default split takes place at 'space'
        ticket = map(lambda t : t.strip(), ticket)   #.strip() removes all whitespace at the start and end 
        ticket = list(filter(lambda t : not t.isdigit(), ticket))   #------RESOLVE----------------#
        if len(ticket) > 0:                                         #how all the non digit numbers are included in ticket[0]
            return ticket[0]
        else: 
            return 'XXX'
    # Extracting dummy variables from tickets:
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)
    status('ticket')

process_ticket()
print(combined.head())

#processing family
def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)
    status('family')

process_family()
print(combined.shape)

combined.drop('PassengerId', inplace=True, axis=1)
print("finally we have the following 68 features to predict survival: \n",combined.head())

##MODELLING##
#There is a wide variety of models to use, from logistic regression to decision trees and more
#sophisticated ones such as random forests and gradient boosted trees.

#scoring function
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

def recover_train_test_target():
    global combined
    train0 = pd.read_csv('C:\\Users\\inspiron\\Desktop\\titanic_survival\\titanic survival train.csv')
    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]
    return train, test, targets

train, test, targets = recover_train_test_target()

#feature selection
#Tree-based estimators can be used to compute feature importances, which in turn can be used to discard irrelevant features.

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')  #n_estimators is the no. of trees in the forest.
clf = clf.fit(train,targets)
output=clf.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('C:\\Users\\inspiron\\Desktop\\titanic_survival\\titanic survival test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
print(df_output)
print("Accuracy score: ",compute_score(clf, train, targets, scoring='accuracy'))

##Lets work on improving accuracy

#calculatig the importance of features
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
#plotting the importance
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))

#Let's now transform our train set and test set in a more compact datasets.
model = SelectFromModel(clf, prefit=True)   #for feature selection
train_reduced = model.transform(train)     #gives different value of train_reduced.shape at differnt times of run (12 or 13 or 14)
print("reduced shape of the train set in new model based on importance of features:\n",train_reduced.shape)
test_reduced = model.transform(test)        #reduces test to selected features
print("reduced shape of the test set in new model based on importance of features:\n",test_reduced.shape)

##  HYPERPARAMETERS TUNING  ##(through grid search algorithm)

#Random Forest are quite handy. They do however come with some parameters to tweak in order to get an optimal model for the prediction task.
# turn run_gs to True if you want to run the gridsearch again.

run_gs = False
if run_gs:
#multiple values for different parameters are given in the parameter_grid for the grid search algo
#to find the best suitable parameters to get the best model.
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [1, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation) #default is 3 fold cross validation

    grid_search.fit(train,targets) 
    model = grid_search

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_)) #gives the best value selected
                                                                  #from the parameter_grid for all
                                                                  #the parameters i.e max_depth, 
                                                                  #n_estimators,max_features, etc.
                                                                  
else: 
    #parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
    #              'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    #model = RandomForestClassifier(**parameters)
    model = RandomForestClassifier(bootstrap= False, min_samples_leaf= 3, n_estimators= 50, 
                  min_samples_split= 10, max_features= 'sqrt', max_depth= 6)
    model.fit(train_reduced, targets)
    
print("Accuracy score: ",compute_score(model, train_reduced, targets, scoring='accuracy'))
output = model.predict(test_reduced).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('C:\\Users\\inspiron\\Desktop\\titanic_survival\\titanic survival test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('titanic survival submission.csv',index=False)

















