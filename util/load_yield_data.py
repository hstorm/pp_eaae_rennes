# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
os.chdir("/home/storm/Research/pp_eaae_rennes")

# %%


def getData():
    # %
    # import yield data from csv
    df = pd.read_csv(os.path.join("data","yield.csv"))
    
    # Get list of crops
    lstCrops = df.columns[df.columns.str.contains("_yield")].str.replace("_yield","")
    
    # Split dfL in train and test set
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # scale yield per crop
    scale_train = {}
    for c in lstCrops:
        scale_train[f'{c}_yield_mean'] = df_train[f'{c}_yield'].mean()
        scale_train[f'{c}_yield_std'] = df_train[f'{c}_yield'].std()
        df_train[f'{c}_yield_scaled'] = (df_train[f'{c}_yield']-scale_train[f'{c}_yield_mean'])/scale_train[f'{c}_yield_std']
        df_test[f'{c}_yield_scaled'] = (df_test[f'{c}_yield']-scale_train[f'{c}_yield_mean'])/scale_train[f'{c}_yield_std']
    # %

    lstSmi25 = list(df.columns[df.columns.str.contains("_25")])
    lstSmi180 = list(df.columns[df.columns.str.contains("_gesamt")])

    # Rearrange data to long format
    lstLong = {'train':[],'test':[]}
    for sDat,dfr in [('train',df_train),('test',df_test)]:
        for c in lstCrops:
            dfi = dfr.loc[:,['NUTS3','year','smi',
                            f'{c}_yield',f'{c}_yield_scaled',
                            f'{c}_bodenzahl',f'{c}_weight']+lstSmi25+lstSmi180]
            dfi.rename(columns={f'{c}_yield':'yield',
                                f'{c}_yield_scaled':'yield_scaled',
                                f'{c}_bodenzahl':'bodenzahl',
                                f'{c}_weight':'weight'},inplace=True)
            dfi['crop'] = c
            lstLong[sDat].append(dfi)
        
    dfL_train = pd.concat(lstLong['train'])
    dfL_test = pd.concat(lstLong['test'])

    # rename crops Mais
    dfL_train['crop'].replace({'Silomais/Grünmais (einschl. Lieschkolbenschrot)':'Mais',
                    'Roggen und Wintermenggetreide':'Roggen'},inplace=True) 
    dfL_test['crop'].replace({'Silomais/Grünmais (einschl. Lieschkolbenschrot)':'Mais',
                    'Roggen und Wintermenggetreide':'Roggen'},inplace=True) 

    # transform str to category
    lstCatCrop = list(dfL_train['crop'].astype('category').cat.categories)
    dfL_train['crop_cat'] = pd.Categorical(dfL_train['crop'], categories=lstCatCrop, ordered=False).codes
    dfL_test['crop_cat'] = pd.Categorical(dfL_test['crop'], categories=lstCatCrop, ordered=False).codes
    
    lstCatNUTS3 = list(dfL_train['NUTS3'].astype('category').cat.categories)
    dfL_train['NUTS3_cat'] = pd.Categorical(dfL_train['NUTS3'], categories=lstCatNUTS3, ordered=False).codes
    dfL_test['NUTS3_cat'] = pd.Categorical(dfL_test['NUTS3'], categories=lstCatNUTS3, ordered=False).codes

    # Drop observations where bodenzahl is zero
    dfL_train = dfL_train.loc[dfL_train['bodenzahl']!=0.,:]
    dfL_test = dfL_test.loc[dfL_test['bodenzahl']!=0.,:]
    
    # scale yield, bodenzahl and smi
    scale_train['bodenzahl_mean'] = dfL_train['bodenzahl'].mean()
    scale_train['bodenzahl_std'] = dfL_train['bodenzahl'].std()

    dfL_train['bodenzahl_scaled'] = (dfL_train['bodenzahl']-scale_train['bodenzahl_mean'])/scale_train['bodenzahl_std']
    dfL_test['bodenzahl_scaled'] = (dfL_test['bodenzahl']-scale_train['bodenzahl_mean'])/scale_train['bodenzahl_std']
    
    # yield_crop = dfL['yield_scaled'].values

    # Drop yield na values
    dfL_train.dropna(subset=['yield'],inplace=True)
    dfL_test.dropna(subset=['yield'],inplace=True)
    # check if there are any na values
    assert dfL_train.isna().sum().sum() == 0
    assert dfL_test.isna().sum().sum() == 0
    
    print('Shape of df train',dfL_train.shape)
    print('Shape of df test',dfL_test.shape)
    print('Train Set: value counts of crop',dfL_train.value_counts('crop'))
    print('Test Set: value counts of crop',dfL_test.value_counts('crop'))

    #%
    return dfL_train, dfL_test, lstCatCrop, lstCatNUTS3, lstSmi25, lstSmi180, scale_train

# %%

if __name__ == "__main__":
    # %%
    dfL_train, dfL_test, lstCatCrop, lstCatNUTS3, lstSmi25, lstSmi180, scale_train = getData()
    
# %%
