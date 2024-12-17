import streamlit as st
import pickle
import numpy as np
import datetime


st.title("Real Estate Price Prediction Model")
st.write("Let's start building!.")

model = pickle.load(open("/workspaces/Real_estate_price_prediction_ml/xgb_realestate.pkl",'rb'))

def ppred(features):
    ip = np.array(features).reshape(1,-1)
    result = model.predict(ip)
    return result [0]

def main():

    AREA = st.selectbox(label="Area",options=["Adyar","Anna Nagar", "Chrompet","KK Nagar","Karapakkam","T Nagar","Velachery"])
    AREA_MAP ={"Adyar":0,"Anna Nagar":1, "Chrompet":2,"KK Nagar":3,"Karapakkam":4,"T Nagar":5,"Velachery":6}
    AREA_VALUE = AREA_MAP[AREA]

    INT_SQFT = st.text_input("SQ.Feet")
    DIST_MAINROAD = st.text_input("Distance from Mainroad")
    N_BEDROOM = st.selectbox("No Of Beedrooms",[1,2,3,4])
    N_BATHROOM = st.selectbox("No Of Bathrooms",[1,2])
    N_ROOM = st.selectbox("Total No Of Rooms",[2,3,4,5,6])
    
    SALE_COND = st.selectbox("Condition",["AbNormal","AdjLand","Family","Normal Sale","Partial"])
    SALE_COND_MAP = {"AbNormal":0,"AdjLand":1,"Family":2,"Normal Sale":3,"Partial":4}
    SALE_COND_VALUE = SALE_COND_MAP[SALE_COND]
    
    PARK_FACIL = st.selectbox("Parking Facility",["Yes","No"])
    PARK_FACIL_MAP = {"Yes":1,"No":0}
    PARK_FACIL_VALUE = PARK_FACIL_MAP[PARK_FACIL]

    BUILDTYPE = st.selectbox("Building Type",["Commercial","House","Others"])
    BUILDTYPE_MAP = {"Commercial":0,"House":1,"Others":2}
    BUILDTYPE_VALUE = BUILDTYPE_MAP[BUILDTYPE]

    UTILITY_AVAIL = st.selectbox("Available Utility",["AllPub", "ELO", "No Sewage"])
    UTILITY_AVAIL_MAP = {"AllPub":0, "ELO":1, "No Sewage":2}
    UTILITY_AVAIL_VALUE = UTILITY_AVAIL_MAP[UTILITY_AVAIL]

    STREET = st.selectbox("Street Access",["Gravel","No Access","Paved"])
    STREET_MAP = {"Gravel":0,"No Access":1,"Paved":2}
    STREET_VALUE = STREET_MAP[STREET]

    MZZONE = st.selectbox("Zone Category",["Agricultural Zone (A)","Commercial Zone (C)","Industrial Zone (I)",
                                           "Residential High-Density Zone (RH)","Residential Low-Density Zone (RL)",
                                           "Residential Medium-Density Zone (RM)"])
    MZZONE_MAP = {"Agricultural Zone (A)":0,"Commercial Zone (C)":1,"Industrial Zone (I)":2,
                                           "Residential High-Density Zone (RH)":3,"Residential Low-Density Zone (RL)":4,
                                           "Residential Medium-Density Zone (RM)":5}
    MZZONE_VALUE = MZZONE_MAP[MZZONE]


    QS_ROOMS = st.slider("Quality Of Rooms", min_value=2.0,max_value = 5.0,value = 2.5,step=0.1)   
    QS_BATHROOM =st.slider("Quality Of Bathrooms",min_value = 2.0, max_value = 5.0,value = 2.5,step =0.1)
    QS_BEDROOM =st.slider("Quality Of Bedrooms",min_value = 2.0, max_value = 5.0,value = 2.5,step =0.1)
    QS_OVERALL = st.slider("Overall Quality",min_value = 2.0, max_value = 5.0,value = 2.5,step =0.1)
    

    REG_FEE = st.text_input("Approximate Registration Budget")
    COMMIS = st.text_input("Approximate Commission Budget")
    sales_date = st.date_input("Sale Date",min_value=datetime.date(1990,1,1),max_value = datetime.date.today())
    year_date_sale = sales_date.year
    month_date_sale = sales_date.month
    day_date_sale = sales_date.day
    week_date_sale = sales_date.isocalendar()[1]

    build_age = st.text_input("Approximate Building Age")
    
    if st.button("Predict"):
        try:
            features=(AREA_VALUE,float(INT_SQFT), float(DIST_MAINROAD), N_BEDROOM, N_BATHROOM,
          N_ROOM,SALE_COND_VALUE,PARK_FACIL_VALUE, BUILDTYPE_VALUE, UTILITY_AVAIL_VALUE,
          STREET_VALUE, MZZONE_VALUE, QS_ROOMS, QS_BATHROOM, 
          QS_BEDROOM,QS_OVERALL, float(REG_FEE), 
          float(COMMIS), int(year_date_sale), int(month_date_sale),int(day_date_sale), week_date_sale, 
          int(build_age))
            prediction = ppred(features)
            st.success(f"The predicted sale price is approximately: ₹{prediction:.2f}")
        except Exception as e:
            st.error(f"Error:, {e}")

if __name__ == "__main__":
    main()


