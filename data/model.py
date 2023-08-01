import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit , cross_val_score


import pickle
import json


matplotlib.rcParams["figure.figsize"]=(20,10)


df1=pd.read_csv("Bengaluru_House_Data.csv")

#print(df1.head())
#df1.shape


#DATA CLEANING




#drop unnecesary columns
df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head


df2=df2.dropna()#drop null records

#dealing with different data bedroom==bhk

df2['size'].unique()
df2['bhki']=df2['size'].apply(lambda x:x.split(' ')[0])
df2['bhk']=df2['bhki'].apply(lambda x:float(x))#typecasting the bhk values from string to float
df2=df2.drop(['bhki'],axis='columns')
#print(df2.head())


def is_float(x):#function to check if values are numbers or range
    try:
        float(x)
    except:
        return False
    return True

#print(df2[~df2['total_sqft'].apply(is_float)].head) #finding the ranges

def convert_range(x): #function to replace ranges with their average
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df3=df2.copy() #making new df and appending the changed range values 
df3['total_sqft']=df3['total_sqft'].apply(convert_range)
#print(df3.loc[30])





#DIMENSIONALITY REDUCTION


df4=df3.copy()#appending new column used in real estate price per sqft
df4['pps']=df4['price']*100000/df4['total_sqft']
df4=df4.drop(['size'],axis='columns')
#print(df4.head())

#print(len(df4.location.unique()))#high dimensionality
df4.location=df4.location.apply(lambda x:x.strip())
location_stats=df4.groupby('location')['location'].agg('count').sort_values(ascending=False)#finding the locations with least data points
#print(location_stats)

#print(len(location_stats[location_stats<=10]))#placing the records with less than 10 datapoints intoother column

other=location_stats[location_stats<=10]
#print(other)
df4.location=df4.location.apply(lambda x:'other' if x in other else x)#converted less than 10 to other
#print(len(df4.location.unique()))



#OUTLIER DETECTION AND REMOVAL

#300SQFT PER BEDROOM IS TYPICAL THRESHOLD  as per domain knowledge

#print(df4[float(df4.total_sqft)/float(df4.bhk)<300].head())
df5=df4[~(df4.total_sqft/df4.bhk<300)]

#print(df5.pps.describe()) #searching for extreme outliers


def rem_out(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):#getting subdf per location
        m=np.mean(subdf.pps)
        sd=np.std(subdf.pps)
        reduced_df=subdf[(subdf.pps>(m-sd))&(subdf.pps<=(m+sd))]#filtering values greater than m-std , less than m+stdand storing in reduced df
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)#appending in df
    return df_out

df6=rem_out(df5)#outliers removed
#print(df6.shape)

def scat(df,location):
    bhk2=df[(df.location==location)&(df.bhk==2)]# aaa location lo aaa bhk unna points
    bhk3=df[(df.location==location)&(df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.pps,color='yellow',label='2BHK',s=50) 
    plt.scatter(bhk3.total_sqft,bhk3.pps,marker='+',color='green',label='3BHK',s=50)
    plt.xlabel("tot_sqft_area")
    plt.ylabel("price per sqft")
    plt.title(location)
    plt.legend()
    plt.show()

#scat(df6,"Rajaji Nagar")#by this we found some 2 bhk of same location are higher than 3bhk , so remove


def rem_locout(df):
    exclude_indices=np.array([])
    for location , location_df in df.groupby('location'):#grouping by bhk
        bhk_stats={}#dict for each type of bhk 
        for bhk ,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={               #per bhk df counting mean , std , count
                'mean':np.mean(bhk_df.pps),
                'std':np.std(bhk_df.pps),
                'count':bhk_df.shape[0]
                }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:#storing in excluding values if pps of 2bhk is less than mean pps of 1bhk 
                exclude_indices=np.append(exclude_indices , bhk_df[bhk_df.pps<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

    
df7=rem_locout(df6)
#print(df7.shape)
#scat(df7,"Rajaji Nagar")#verification

plt.hist(df7.pps,rwidth=0.8)
plt.xlabel("pps")
plt.ylabel("count")
#plt.show()

df7.bath.unique() 
plt.hist(df7.bath,rwidth=0.8)
plt.xlabel("no of bathrooms")
plt.ylabel("count")
#plt.show()

#print(df7[df7.bath > (df7.bhk+2)])#we cant have no of baths > no of bed+2
df8=df7[df7.bath<(df7.bhk+2)]
#print(df8.shape)
#print(df8.head)

df8=df8.drop(['pps'],axis='columns')#since outlier removal is completed we drop this unnecessary pps column
#print(df8.head)






#converting the locations into numeric values using one hot encoding
dummies=pd.get_dummies(df8.location)
df9=pd.concat([df8,dummies.drop('other',axis='columns')],axis='columns')

df9=df9.drop('location',axis='columns')
print(df9.shape)

x=df9.drop('price',axis='columns') #now x consists of only independent variables
y=df9.price


x_train,x_test ,y_train ,y_test= train_test_split(x,y,test_size=0.2,random_state=10) #splitting data to train and test

lr=LinearRegression()#lasso , decesion tree also tried butless accuracy
lr.fit(x_train,y_train)
#print(lr.score(x_test,y_test))#testing the sample with linear regressions

cv=ShuffleSplit(n_splits=5 , test_size=0.2 , random_state=0)#shuffling data for finding avg accuracy
#print(cross_val_score(LinearRegression(),x,y,cv=cv))





def prediction(location , sqft , bath , bhk):
    loc_index=np.where(x.columns==location)[0][0]

    X=np.zeros(len(x.columns))
    X[0]=sqft
    X[1]=bath
    X[2]=bhk
    if loc_index>=0:
        X[loc_index]=1

    return lr.predict([X])[0]    

#print('the price in Lakhs is :',prediction('1st Phase JP Nagar',1000 , 2,2))







#exporting model to a pickle file
with open('banglore_home_price.pickle','wb')as f:
    pickle.dump(lr,f)

columns={
    'data_columns' : [col.lower() for col in x.columns]}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
