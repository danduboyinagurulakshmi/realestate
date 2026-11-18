from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.real_estate_pipeline import load_model_bundle, predict_property, save_recommendation_summary


DATA_PATH = Path(__file__).resolve().parent / "india_housing_prices.csv"


@st.cache_resource
def get_model_bundle():
    return load_model_bundle(Path(__file__).resolve().parent / "models")


@st.cache_data
def get_market_summary():
    return save_recommendation_summary(DATA_PATH, Path(__file__).resolve().parent / "plots" / "eda_summary.json")


st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")


st.title("Real Estate Investment Advisor")
st.caption("Predict whether a property is a good investment and forecast its 5-year value.")

with st.sidebar:
    st.header("Property Inputs")
    state = st.selectbox("State", [
        "Tamil Nadu", "Maharashtra", "Punjab", "Rajasthan", "West Bengal", "Chhattisgarh", "Delhi",
        "Karnataka", "Gujarat", "Telangana", "Uttar Pradesh", "Haryana", "Kerala", "Andhra Pradesh",
        "Bihar", "Madhya Pradesh", "Odisha", "Assam", "Jharkhand", "Goa"
    ])
    city = st.selectbox("City", [
        "Chennai", "Pune", "Ludhiana", "Jodhpur", "Jaipur", "Durgapur", "Bilaspur", "New Delhi",
        "Bengaluru", "Ahmedabad", "Hyderabad", "Lucknow", "Gurugram", "Thiruvananthapuram", "Visakhapatnam",
        "Patna", "Bhopal", "Cuttack", "Silchar", "Ranchi", "Panaji"
    ])
    locality = st.text_input("Locality", value="Locality_101")
    property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
    size_sqft = st.number_input("Size in Sq Ft", min_value=200.0, max_value=10000.0, value=1800.0, step=50.0)
    price_in_lakhs = st.number_input("Current Price (Lakhs)", min_value=10.0, max_value=1000.0, value=180.0, step=5.0)
    year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2015, step=1)
    furnished_status = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
    floor_no = st.number_input("Floor No", min_value=1, max_value=50, value=8, step=1)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=60, value=12, step=1)
    nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=20, value=6, step=1)
    nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=20, value=4, step=1)
    public_transport = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
    parking_space = st.selectbox("Parking Space", ["No", "Yes"])
    security = st.selectbox("Security", ["No", "Yes"])
    amenities = st.text_input("Amenities", value="Gym, Pool, Clubhouse")
    facing = st.selectbox("Facing", ["North", "South", "East", "West"])
    owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
    availability_status = st.selectbox("Availability Status", ["Ready_to_Move", "Under_Construction"])

    property_payload = {
        "ID": 99999,
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_in_Lakhs": price_in_lakhs,
        "Price_per_SqFt": price_in_lakhs * 100000 / size_sqft,
        "Year_Built": year_built,
        "Furnished_Status": furnished_status,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": max(0, 2025 - year_built),
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": public_transport,
        "Parking_Space": parking_space,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability_status,
    }

    if st.button("Predict Investment Potential"):
        model_bundle = get_model_bundle()
        result = predict_property(property_payload, model_bundle)
        st.session_state["prediction"] = result

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    invest_label = "Good Investment" if prediction["good_investment"] else "High Risk"
    confidence = prediction["confidence"]
    projected_price = prediction["projected_5_year_price_lakhs"]
    expected_appreciation = prediction["expected_appreciation"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommendation", invest_label)
    col2.metric("Confidence", f"{confidence * 100:.1f}%")
    col3.metric("5-Year Price", f"₹{projected_price:,.2f} L")

    st.subheader("Investment Summary")
    st.write(
        f"This property is projected to grow by {expected_appreciation:.1f}% over five years and is classified as a {invest_label.lower()} with {confidence * 100:.1f}% confidence."
    )

st.subheader("Market Trends")
summary = get_market_summary()
market_col1, market_col2, market_col3 = st.columns(3)
market_col1.metric("Average Price", f"₹{summary['price_summary']['mean']:.2f} L")
market_col2.metric("Median Price", f"₹{summary['price_summary']['median']:.2f} L")
market_col3.metric("Good Investment Share", f"{summary['good_investment_share']:.1f}%")

sample_df = pd.read_csv(DATA_PATH)
city_prices = sample_df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10).reset_index()
fig = px.bar(city_prices, x="City", y="Price_in_Lakhs", title="Average property price by city")
st.plotly_chart(fig, use_container_width=True)

scatter = px.scatter(sample_df.sample(min(1500, len(sample_df))), x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type", title="Size vs Price")
st.plotly_chart(scatter, use_container_width=True)
