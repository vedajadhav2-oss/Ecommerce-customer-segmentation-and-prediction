import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

#Load the Cleaned Dataset
data=pd.read_csv(r"C:\Users\jveda\Desktop\BIA\Capstone pro\Streamlit application\Cleaned_data.csv")

#Convert Invoicedate to datetime
data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])

print("Data Load successfully")

#RFM Segmentation
rfm=pd.read_csv(r"C:\Users\jveda\Desktop\BIA\Capstone pro\Streamlit application\rfm.csv")
print(rfm.head(2))

#Rfm scaling
scaler=StandardScaler()
rfm_scaled=scaler.fit_transform(rfm[['Recency','Frequency','Monetarry']])
print("RFM Scaling completed")

# TRain KMeans model
kmeans=KMeans(n_clusters=4,random_state=42)
rfm['KMeans_cluster']= kmeans.fit_predict(rfm_scaled)
print("KMeans model trained successfully")
print(rfm[['Recency','Frequency','Monetarry','KMeans_cluster']].head(2))
# Save the KMeans model
joblib.dump(kmeans,r"C:\Users\jveda\Desktop\BIA\Capstone pro\Streamlit application\kmeans_model.pkl")
print("KMeans model saved successfully")        
joblib.dump(scaler,r"C:\Users\jveda\Desktop\BIA\Capstone pro\Streamlit application\scaler_model.pkl")
print("Scaler model saved successfully")

# Analyze clusters
cluster_analysis=rfm.groupby('KMeans_cluster')[['Recency','Frequency','Monetarry']].mean()
print("Cluster Analysis:")
print(cluster_analysis)

# Cluster naming mapping
cluster_names={
    0:'High-Value Customers',
    1:'Regular Customers',
    2:'VIP Customers',
    3:'Low-Value Customers'
}       
rfm['Customer_Segment']=rfm['KMeans_cluster'].map(cluster_names)
print(rfm[['KMeans_cluster','Customer_Segment']].head(2))
rfm.to_csv(r"C:\Users\jveda\Desktop\BIA\Capstone pro\Streamlit application\rfm_with_segments.csv",index=False)
print("RFM with segments saved successfully")

# Train Random Forest Classifier to predict customer segments
X=rfm[['Recency','Frequency','Monetarry']]
y=rfm['Customer_Segment']
rf_classifier=RandomForestClassifier(n_estimators=100,random_state=42)
rf_classifier.fit(X,y)  
print("Random Forest Classifier trained successfully")
# Save the Random Forest model
joblib.dump(rf_classifier,r"C:\Users\jveda\Desktop\BIA\Capstone pro\Streamlit application\rf_classifier_model.pkl")
print("Random Forest model saved successfully") 