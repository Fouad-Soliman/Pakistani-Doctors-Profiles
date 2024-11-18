import pandas as pd 
import numpy as np 
import joblib
from sklearn.metrics import r2_score,mean_squared_error

space=""
Nospace=""
QualSeprator=","
doctors_rank={
    "Prof":4,
    "Assoc":3,
    "Asst":2,
    "Dr":1
}

input_df=pd.read_csv(r"DoctorFeePrediction_test.csv")
predict_df=pd.read_csv(r"TrainColumnsReg.csv")

def get_title(x):
    return x[0]

def join_special(word_list:list):
    
    for i,word in enumerate(word_list):
        word_list[i]=str(word_list[i]).strip()
        word_list[i]=Nospace.join(word_list[i])


        if word_list[i]=="" or word_list[i]==" ":
            found=True
            
            word_list.remove(word_list[i])
            continue

    return QualSeprator.join(word_list[:])

def specialLen(word_list:list):
    
    for i,word in enumerate(word_list):
        word_list[i]=str(word_list[i]).strip()
        word_list[i]=Nospace.join(word_list[i])

       
        if word_list[i]=="" or word_list[i]==" ":
            found=True
            
            word_list.remove(word_list[i])
            continue

    return len(word_list)

def join_qual(word_list:list):

    removeing_index=[]
    found=False
    for i,word in enumerate(word_list):
        Strip_Split=str(word_list[i]).strip().split()
        for ind in range (len(Strip_Split)):
            Strip_Split[ind]=Strip_Split[ind].strip()
        word_list[i]=space.join(Strip_Split)

        ## There are some words that can not be handled
        ## ['M.B.B.S', 'F.C.P.S', '(NeuroSurgery)I.T.F(UnitedKingdom)'] => ['MBBS',"FCPS"]
        ## We guess that this can be a good technique to handle different text problems

        # if "." in word_list[i]:
        #     print(word_list[i])
        #     print(word_list)
        if word_list[i]=="" or word_list[i]==" ":
            found=True
            removeing_index.append(word_list[i])
            
            continue

    if found:
        for ind in removeing_index:
            word_list.remove(ind)
    
    

    return QualSeprator.join(word_list[:])

def getLen(word_list:list):
    
    for i,word in enumerate(word_list):
        word_list[i]=str(word_list[i]).strip().split()
        word_list[i]=space.join(word_list[i])

        ## There are some words that can not be handled
        ## ['M.B.B.S', 'F.C.P.S', '(NeuroSurgery)I.T.F(UnitedKingdom)'] => ['MBBS',"FCPS"]
        ## We guess that this can be a good technique to handle different text problems

        # if "." in word_list[i]:
        #     print(word_list[i])
        #     print(word_list)
        if word_list[i]=="" or word_list[i]==" ":
            found=True    
            word_list.remove(word_list[i])
            continue
    
    return len(word_list)


def Preprocess(test_df:pd.DataFrame):

    test_df["Doctor Name"].fillna("Dr",inplace=True)
    
    test_df["City"].fillna("NULL",inplace=True)

    test_df["Experience(Years)"].fillna(10,inplace=True)
    
    test_df["Total_Reviews"].fillna(8,inplace=True)
    
    test_df["Patient Satisfaction Rate(%age)"].fillna(98.0,inplace=True)
    
    test_df["Avg Time to Patients(mins)"].fillna(14.0,inplace=True)
    
    test_df["Wait Time(mins)"].fillna(11.0,inplace=True)

    test_df["Doctors Title"]=test_df["Doctor Name"].str.lower().str.strip().str.split(".").apply(lambda x : get_title(x)).str.capitalize()
    
    test_df["Doctors Title"]=test_df["Doctors Title"].map(doctors_rank)

    test_df["Doctors Title"].fillna(1,inplace=True)
    

    test_df["Specialization"].fillna("NULL",inplace=True)
    test_df["Specialization"]=test_df["Specialization"].str.lower().str.strip().str.split(",").apply(lambda x : join_special(x))
   
    test_df["Specialization_No"]=np.where(test_df["Specialization"]=="null",0,test_df["Specialization"].str.strip().str.split(",").apply(lambda x : specialLen(x)))
   
    
    test_df["Doctor Qualification"].fillna("NULL",inplace=True)
    test_df["Doctor Qualification"]=test_df["Doctor Qualification"].str.lower().str.replace("*","").str.replace(']','}').str.replace('[','{').str.replace('(','{').str.replace(')',"}").str.strip().str.split(",").apply(join_qual)
   
    test_df["Doctor_Qualification_No"]=np.where(test_df["Doctor Qualification"]=="null",0,test_df["Doctor Qualification"].str.lower().str.replace("*","").str.replace(']','}').str.replace('[','{').str.replace('(','{').str.replace(')',"}").str.strip().str.split(",").apply(getLen))
    
    enc = pd.get_dummies(test_df['City'], prefix='City')
    
    enc1=test_df.Specialization.str.get_dummies(sep=',').add_prefix('Specialization_')
    
    enc2=test_df['Doctor Qualification'].str.get_dummies(sep=',').add_prefix('Qualification_')
    
    test_df[['Experience(Years)', 'Total_Reviews']] = np.log(test_df[['Experience(Years)', 'Total_Reviews']]+1)

    test_df=pd.concat([test_df.drop(columns=['City','Specialization','Doctor Qualification']),enc,enc1,enc2],axis=1)
    
    test_df.reset_index(drop=True,inplace=True)

    for col in predict_df.columns:
        if col in test_df.columns:
            predict_df[col] = test_df[col]
        else:
            predict_df[col] = 0
    



y_test= np.sqrt(input_df["Fee(PKR)"])
input_df.drop(columns="Fee(PKR)",axis=1,inplace=True)

Preprocess(input_df)


model = joblib.load('RFRegression.pkl')

pred=model.predict(predict_df)


print("Random Forest Regressor MSE : ", mean_squared_error(y_test**2,pred**2))
print("Random Forest Regressor R2 : ", r2_score(y_test**2,pred**2))

