import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#csv
train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')

train = train_df.copy()
#test = test_df.copy()
train.isnull().sum()


# Mapping -> ordinal categorical data !!!
import category_encoders as encoders
train3 = train.copy()

cat_ordinal=['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'HeatingQC','KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual' ,'GarageCond',
             'PavedDrive','PoolQC','Fence']

enc = encoders.OrdinalEncoder(mapping = [{'col': 'LotShape', 'mapping': {'IR3':0, 'IR2':1,'IR1':2,'Reg':3}},
                                         {'col': 'LandSlope', 'mapping': {'Gtl':3, 'Mod':2, 'Sev':1}},
                                         {'col': 'ExterQual', 'mapping': {'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}},
                                         {'col': 'ExterCond', 'mapping': {'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}},
                                         {'col': 'BsmtQual', 'mapping': {'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}},
                                         {'col': 'BsmtCond', 'mapping': {'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}},
                                         {'col': 'BsmtExposure', 'mapping': {'Gd':4, 'Av':3, 'Mn':2,'No':1, 'NA':0}},
                                         {'col': 'BsmtFinType1', 'mapping': {'GLQ':6,'ALQ':5,'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0}},
                                         {'col': 'BsmtFinType2', 'mapping': {'GLQ':6,'ALQ':5,'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0}},
                                         {'col': 'HeatingQC', 'mapping': {'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}},
                                         {'col': 'KitchenQual', 'mapping': {'Ex':4,'Gd':3, 'TA':2, 'Fa':1, 'Po':0}},
                                         {'col': 'Functional', 'mapping': {'Typ':7,'Min1':6, 'Min2':5, 'Mod':4, 'Maj1':3, 'Maj2':2, 'Sev':1,'Sal':0}},
                                         {'col': 'FireplaceQu', 'mapping': {'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}},
                                         {'col': 'GarageFinish', 'mapping': {'Fin':3, 'RFn':2, 'Unf':1, 'NA':0}},
                                         {'col': 'GarageQual', 'mapping': {'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}},
                                         {'col': 'GarageCond', 'mapping': {'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}},
                                         {'col': 'PavedDrive', 'mapping': {'Y':2, 'P':1, 'N':0}},
                                         {'col': 'PoolQC', 'mapping': {'Ex':4,'Gd':3, 'TA':2, 'Fa':1,'NA':0}},
                                         {'col': 'Fence', 'mapping': {'GdPrv':4,'MnPrv':3, 'GdWo':2, 'MnWw':1,'NA':0}},
                                        ])

train3=enc.fit_transform(train3)
for col in cat_ordinal:
    train3[col]=train3[col].replace(-1,0).astype(int)
    #train3[col]= train3[col].astype(int)
    #print(train3[col]) #.astype(int)
    
for col in cat_ordinal:
    count= train3[col].isnull().sum(axis=0)
    print(train3[col]) 

print(train3) 

#------------------------------------------------------------------------------------------------------------------------------
# one-hot encoding on nominal featurea  (W5 lecture) --> nominal categorical data --> should encode on every nominal features

from sklearn.preprocessing import LabelEncoder

cat_nominal=['MSSubClass','MSZoning','Street','Alley','LandContour','Utilities','LotConfig','Neighborhood','Condition1',
            'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','CentralAir',
            'Electrical','GarageType','SaleType','SaleCondition','Heating','MiscFeature','MasVnrType']


NA_nominal=[]

train1 = train.copy()

print("features with NA data field :")
for col in cat_nominal:
    count= train1[col].isnull().sum(axis=0)
   
    if count != 0 :
        print(col,count)
        NA_nominal.append(col)
        #identify feature with NA value
        '''
        Alley 1369 
        Electrical 1
        GarageType 81
        MiscFeature 1406 
        
        '''
print(NA_nominal)
train2 = train.copy()

train2=pd.get_dummies(train2, columns=NA_nominal)

print(train2.dtypes)

# NEXT

 '''  condition1 & 2 can be merge perhaps, some have 1 , some have 2 condition
    Exterior1st & Exterior2nd can be merged '''
#------------------------------------------------------------------------------------------------------------------------------
   # --> numerical data ??
numerical=['LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
            '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
            'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
            '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold']
