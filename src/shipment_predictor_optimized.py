import xgboost as xgb
import pandas as pd
import numpy as np
import json
import csv
import pickle
from geopy.distance import geodesic
from joblib import load
from src.codes import ship_mode_codes, scac_2_values_codes, carrier_service_code_1_values_codes, \
carrier_service_code_2_values_codes, seller_organization_code_values_codes, item_id_values_codes, ship_node_key_values_codes, \
holidays_pd_datetime



class ShipmentPredictorOptimized:

    model = None
    
    def __init__(self, model_path, scalar_file) -> None:
        self.model = pickle.load(open(model_path, "rb"))
        self.scaler = pickle.load(open(scalar_file, "rb"))
        self.zip_to_co_ordinates = self._load_co_ordinates(
            filepath="data/us-zip-codes.csv"
        )
    
    #clean values from column
    def clean_column_value(self, df):
        df['ITEM_ID'] = df['ITEM_ID'].str.split("-", expand=True)[0]
        df['SHIP_MODE'] = df['SHIP_MODE'].str.strip()
        df['SCAC_2'] = df['SCAC_2'].str.strip()
        df['CARRIER_SERVICE_CODE_1'] = df['CARRIER_SERVICE_CODE_1'].str.strip()
        df['CARRIER_SERVICE_CODE_2'] = df['CARRIER_SERVICE_CODE_2'].str.strip()
        df['ZIP_CODE'] = df['ZIP_CODE'].str.split("-", expand=True)[0]
        df['ZIP_CODE_1'] = df['ZIP_CODE_1'].str.strip()
        df['ORDER_DATE'] = df['ORDER_DATE'].str.strip()
        df['ACTUAL_SHIPMENT_DATE'] = df['ACTUAL_SHIPMENT_DATE'].str.strip()
        df['SELLER_ORGANIZATION_CODE_1'] = df['SELLER_ORGANIZATION_CODE_1'].str.strip()
        df['SHIPNODE_KEY_1'] = df['SHIPNODE_KEY_1'].str.strip()
    



    #Update datatypes
    def update_datatype(self, df):
        df['ORDERED_QTY'] = df['ORDERED_QTY'].astype('float64')
        df['ITEM_WEIGHT'] = df['ITEM_WEIGHT'].astype('float64')
        df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'],format="%Y-%m-%d %H:%M:%S.%f")
        df['ACTUAL_SHIPMENT_DATE'] = pd.to_datetime(df['ACTUAL_SHIPMENT_DATE'],format="%Y-%m-%d %H:%M:%S.%f")

    #Convert String 'NaN' to np.nan
    def convert_nan(self, df):
        df.replace('NaN', float(np.nan), regex=True,inplace=True)


    def extract_date_features(self, data):
        data["day"] = data.ORDER_DATE.dt.day.astype(float)
        data["month"] = data.ORDER_DATE.dt.month.astype(float)
        data["quarter"] = data.ORDER_DATE.dt.quarter.astype(float)
        data["year"] = data.ORDER_DATE.dt.year.astype(float)
        data['day_of_week'] = data.ORDER_DATE.dt.day_of_week.astype(float)
        data["is_month_start"] = data.ORDER_DATE.dt.is_month_start.astype(float)
        data["is_month_end"] = data.ORDER_DATE.dt.is_month_end.astype(float)
        data["is_quarter_start"] = data.ORDER_DATE.dt.is_quarter_start.astype(float)
        data["is_quarter_end"] = data.ORDER_DATE.dt.is_quarter_end.astype(float)
        data["is_year_start"] = data.ORDER_DATE.dt.is_year_start.astype(float)
        data["is_year_end"] = data.ORDER_DATE.dt.is_year_end.astype(float)
        data['is_weekend'] = np.where(data['day_of_week'].isin([5,6]),1,0)
        data['is_holiday'] = np.where(pd.to_datetime(data.ORDER_DATE.dt.date).isin(holidays_pd_datetime), 1, 0)

    def _load_co_ordinates(self, filepath: str):
        #Load co-ordinates
        zip_to_co_ordinates = {}
        with open(filepath, 'r', encoding="utf-8") as ifile:
            reader = csv.DictReader(ifile, delimiter=",")
            for row in reader:
                zip_to_co_ordinates[str(row['ZIP']).zfill(5)] = [row['LAT'], row['LNG']]
        return zip_to_co_ordinates
    
    def lookup_co_ordinates(self, zip_code: str):
        zip_code = str(zip_code).zfill(5)
        return self.zip_to_co_ordinates[zip_code] if zip_code in self.zip_to_co_ordinates else [0.0, 0.0]
    
    def assign_lat_lng(self, data):
        data['FROM_LAT'] = data['ZIP_CODE'].apply(lambda x: self.lookup_co_ordinates(x)[0])
        data['FROM_LNG'] = data['ZIP_CODE'].apply(lambda x: self.lookup_co_ordinates(x)[1])
        data['TO_LAT'] = data['ZIP_CODE_1'].apply(lambda x: self.lookup_co_ordinates(x)[0])
        data['TO_LNG'] = data['ZIP_CODE_1'].apply(lambda x: self.lookup_co_ordinates(x)[1])

    #Calculate distance between from & to location
    def calculate_distance(self, df):
        df['distance'] = np.zeros(len(df))
        from_coordinates = df[['FROM_LAT','FROM_LNG']].to_numpy()
        to_location_coordinates = df[['TO_LAT','TO_LNG']].to_numpy()
        df['distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in zip(from_coordinates, to_location_coordinates)])
        df['distance'] = df['distance'].astype("str").str.extract('(\d+)').astype("int64")
    
    #Update datatypes
    def update_datatype_lat_long(self, df):
        df['FROM_LAT'] = df['FROM_LAT'].astype('float64')
        df['FROM_LNG'] = df['FROM_LNG'].astype('float64')
        df['TO_LAT'] = df['TO_LAT'].astype('float64')
        df['TO_LNG'] = df['TO_LNG'].astype('float64')


    def calculate_time(self, data):
        # calculate time diff from time stamp
        data['time_taken_mins'] = ((data['ACTUAL_SHIPMENT_DATE'] - data['ORDER_DATE']).dt.seconds.div(60).astype(int) + (data['ACTUAL_SHIPMENT_DATE'] - data['ORDER_DATE']).dt.days.multiply(1440).astype(int))
    

    def encode_categorial_columns(self, data):
        data['ITEM_ID'] = data['ITEM_ID'].apply(lambda x: item_id_values_codes[x]).astype("float")
        data['SHIP_MODE'] = data['SHIP_MODE'].apply(lambda x: ship_mode_codes[x]).astype("float")
        data['SCAC_2'] = data['SCAC_2'].apply(lambda x: scac_2_values_codes[x]).astype("float")
        data['CARRIER_SERVICE_CODE_1'] = data['CARRIER_SERVICE_CODE_1'].apply(lambda x: carrier_service_code_1_values_codes[x]).astype("float")
        data['CARRIER_SERVICE_CODE_2'] = data['CARRIER_SERVICE_CODE_2'].apply(lambda x: carrier_service_code_2_values_codes[x]).astype("float")
        data['SELLER_ORGANIZATION_CODE_1'] = data['SELLER_ORGANIZATION_CODE_1'].apply(lambda x: seller_organization_code_values_codes[x]).astype("float")
        data['SHIPNODE_KEY_1'] = data['SHIPNODE_KEY_1'].apply(lambda x: ship_node_key_values_codes[x]).astype("float")




    # Specify data types for each column
    dtype_mapping = {
        "ITEM_ID": object,
        "ORDERED_QTY": int,
        "SHIP_MODE": object,
        "SCAC_2": object,
        "CARRIER_SERVICE_CODE_1": object,
        "CARRIER_SERVICE_CODE_2": object,
        "ZIP_CODE": object,
        "ZIP_CODE_1": object,
        "ORDER_DATE": object,
        "ACTUAL_SHIPMENT_DATE": object,
        "SELLER_ORGANIZATION_CODE_1": object,
        "ITEM_WEIGHT": float,
        "SHIPNODE_KEY": object,
        "SHIPNODE_KEY_1": object
    }

    def process_request(self, input_request: dict) -> dict:
        in_cols = {col: [input_request[col]] for col, val in self.dtype_mapping.items()}
        input_df = pd.DataFrame(in_cols)
        for col in self.dtype_mapping:
            input_df[col] = input_df[col].astype(self.dtype_mapping[col])
        
        self.clean_column_value(input_df)
        self.update_datatype(input_df)
        self.convert_nan(input_df)

        # Feature engineering
        self.extract_date_features(input_df)
        self.assign_lat_lng(input_df)
        self.update_datatype_lat_long(input_df)
        self.calculate_distance(input_df)
        self.calculate_time(input_df)
        self.encode_categorial_columns(input_df)
        # input_df = self.drop_order_date(input_df)
        input_df = input_df.drop(['ZIP_CODE', 'ZIP_CODE_1', 'ORDER_DATE',  'ACTUAL_SHIPMENT_DATE', 'SHIPNODE_KEY', 'is_holiday'], axis=1)
        print('original', input_df['time_taken_mins'])
        input_df = input_df.drop(['time_taken_mins'], axis=1)
        input_df_numpy = input_df.values
        input_df_numpy_scaled = self.scaler.transform(input_df_numpy)
        y_pred = self.model.predict(input_df_numpy_scaled)
        time = y_pred[0]
        print("Predicted in minutes: "+str(time))
        days = time // 1440     
        leftover_minutes = time % 1440
        hours = leftover_minutes // 60
        mins = time - ((days*1440) + (hours*60))
        return {
            "Predicted Shipment time": str(days) + " days, " + str(hours) + " hours, " + str(mins) +  " mins. "
        }