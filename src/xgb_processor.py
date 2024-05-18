import xgboost as xgb
import pandas as pd
import numpy as np
import json
from geopy.distance import geodesic
from joblib import load

from sklearn.preprocessing import LabelEncoder, StandardScaler 
class XGBProcessor:

  model = None
  def __init__(self, model_path, scalar_file) -> None:
    self.model = xgb.XGBRegressor(n_estimators=20,max_depth=9)
    # self.model = xgb.Booster()
    self.model.load_model(model_path)
    self.scaler = load(scalar_file)
    
    # Fit the scaler on the training data
    
  
  #Update Column Names
  def update_column_name(self,df):
    #Renaming Weatherconditions column
    df.rename(columns={'Weatherconditions': 'Weather_conditions'},inplace=True)

  def update_datatype(self, df):
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('float64')
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype('float64')
    df['multiple_deliveries'] = df['multiple_deliveries'].astype('float64')
    df['Order_Date']=pd.to_datetime(df['Order_Date'],format="%d-%m-%Y")

  
  #Convert String 'NaN' to np.nan
  def convert_nan(self, df):
      df.replace('NaN', float(np.nan), regex=True,inplace=True)



  #Handle null values
  def handle_null_values(self, df):
      df['Delivery_person_Age'].fillna(np.random.choice(df['Delivery_person_Age']), inplace=True)
      df['Weather_conditions'].fillna(np.random.choice(df['Weather_conditions']), inplace=True)
      df['City'].fillna(df['City'].mode()[0], inplace=True)
      df['Festival'].fillna(df['Festival'].mode()[0], inplace=True)
      df['multiple_deliveries'].fillna(df['multiple_deliveries'].mode()[0], inplace=True)
      df['Road_traffic_density'].fillna(df['Road_traffic_density'].mode()[0], inplace=True)
      df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)

  def extract_date_features(self, data):
    data["day"] = data.Order_Date.dt.day
    data["month"] = data.Order_Date.dt.month
    data["quarter"] = data.Order_Date.dt.quarter
    data["year"] = data.Order_Date.dt.year
    data['day_of_week'] = data.Order_Date.dt.day_of_week.astype(int)
    data["is_month_start"] = data.Order_Date.dt.is_month_start.astype(int)
    data["is_month_end"] = data.Order_Date.dt.is_month_end.astype(int)
    data["is_quarter_start"] = data.Order_Date.dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data.Order_Date.dt.is_quarter_end.astype(int)
    data["is_year_start"] = data.Order_Date.dt.is_year_start.astype(int)
    data["is_year_end"] = data.Order_Date.dt.is_year_end.astype(int)
    data['is_weekend'] = np.where(data['day_of_week'].isin([5,6]),1,0)


  #Calculate Time Differnce 
  def calculate_time_diff(self, df):
      # Find the difference between ordered time & picked time
      df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
      df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])
      
      df['Time_Order_picked_formatted'] = df['Order_Date'] + np.where(df['Time_Order_picked'] < df['Time_Orderd'], pd.DateOffset(days=1), pd.DateOffset(days=0)) + df['Time_Order_picked']
      df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']

      # Convert the datetime columns to pandas datetime objects if they are not already
      df['Time_Order_picked_formatted'] = pd.to_datetime(df['Time_Order_picked_formatted'])
      df['Time_Ordered_formatted'] = pd.to_datetime(df['Time_Ordered_formatted'])
      
      df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60
      
      # Handle null values by filling with the median
      df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
      
      # Drop all the time & date related columns
      df.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)


  #Calculate distance between restaurant location & delivery location
  def calculate_distance(self, df):
      df['distance']=np.zeros(len(df))
      restaurant_coordinates=df[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
      delivery_location_coordinates=df[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()
      df['distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in zip(restaurant_coordinates, delivery_location_coordinates)])
      df['distance']= df['distance'].astype("str").str.extract('(\d+)').astype("int64")

  def label_encoding(self, df):
    categorical_columns = df.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()
    df[categorical_columns] = df[categorical_columns].apply(lambda col: label_encoder.fit_transform(col))


  #Extract relevant values from column
  def extract_column_value(self, df):
      #Extract time and convert to int: This not required in inference
      # df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: int(x.split(' ')[1].strip()))
      #Extract Weather conditions
      df['Weather_conditions'] = df['Weather_conditions'].apply(lambda x: x.split(' ')[1].strip())
      #Extract city code from Delivery person ID
      df['City_code']=df['Delivery_person_ID'].str.split("RES", expand=True)[0]

  #Drop Columns which won't be use for building model
  def drop_columns(self, df):
      df.drop(['ID','Delivery_person_ID'],axis=1,inplace=True)

  def process_request(self, input_request: dict) -> dict:


    # Specify data types for each column
    dtype_mapping = {
        "ID": object,
        "Delivery_person_ID": object,
        "Delivery_person_Age": object,
        "Delivery_person_Ratings": object,
        "Restaurant_latitude": float,
        "Restaurant_longitude": float,
        "Delivery_location_latitude": float,
        "Delivery_location_longitude": float,
        "Order_Date": object,
        "Time_Orderd": object,
        "Time_Order_picked": object,
        "Weatherconditions": object,
        "Road_traffic_density": object,
        "Vehicle_condition": int,
        "Type_of_order": object,
        "Type_of_vehicle": object,
        "multiple_deliveries": object,
        "Festival": object,
        "City": object
    }

    

    in_cols = {col: [input_request[col]] for col, val in dtype_mapping.items()}
    input_df = pd.DataFrame(in_cols)
    for col in dtype_mapping:
       input_df[col] = input_df[col].astype(dtype_mapping[col])
    self.update_column_name(input_df)
    self.extract_column_value(input_df)
    self.drop_columns(input_df)
    self.update_datatype(input_df)
    self.convert_nan(input_df)
    self.handle_null_values(input_df)
    self.extract_date_features(input_df)
    self.calculate_time_diff(input_df)
    self.calculate_distance(input_df)
    self.label_encoding(input_df)
    input_df_numpy = input_df.values
    input_df_numpy_scaled = self.scaler.transform(input_df_numpy)
    y_pred = self.model.predict(input_df_numpy_scaled)
    #https://stackoverflow.com/questions/63842190/fit-transform-vs-transform-when-doing-inference
    return {
      "Time Taken in Mins": str(y_pred[0] - 10)
    }


