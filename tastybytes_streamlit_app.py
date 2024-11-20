
# Import Snowflake modules
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark import Window

# Import Snowflake Modeling API
from snowflake.ml.registry import Registry

# Import libraries for Streamlit app
import streamlit as st

from snowflake.snowpark.context import get_active_session
session = get_active_session()

st.title("Select a city to visualize top 20 locations on the map")

selected_city_map = st.text_input("Enter the city 👇")
snowpark_df = session.table("FROSTBYTE_TASTY_BYTES.SCHEMA_MICHELLE.SHIFT_SALES_ALL_FEATURES")

@st.cache_resource
def get_model_version():
    native_registry = Registry(session, database_name="FROSTBYTE_TASTY_BYTES", schema_name="SCHEMA_MICHELLE")
    m = native_registry.get_model("linear_regression")
    return m.default

# Get the date to predict
date_tomorrow = snowpark_df.filter(F.col("shift_sales").is_null()).select(F.min("date")).collect()[0][0]

# Filter to tomorrow's date and the morning shift in {{ selected_city_map }}
location_predictions_df = snowpark_df.filter((F.col("date") == date_tomorrow) 
                                             & (F.col("shift_ohe_AM") == 1) 
                                             & (F.col("city")==selected_city_map))

mv = get_model_version()

# Get predictions
location_predictions_df = mv.run(function_name="predict", X=location_predictions_df).select(
    "city",
    "location_id", 
    "latitude", 
    "longitude",
    "prediction"
)

window = Window.partitionBy(location_predictions_df['city']).orderBy(location_predictions_df['prediction'].desc())
filtered_df = location_predictions_df.select(
    "city",
    "location_id", 
    "latitude", 
    "longitude",
    "prediction",
    F.rank().over(window).alias('rank')).filter(F.col('rank') <= 20)
                                               
# Pull location predictions into a pandas DataFrame
predictions_df = filtered_df.to_pandas()

st.map(predictions_df,
    latitude='latitude',
    longitude='longitude',
    size=1000
)

predictions_df



