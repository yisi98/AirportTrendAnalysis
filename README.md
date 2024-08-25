# Welcome to my pet project on Airport Trend Analysis

### I want to use the kaggle dataset on airport and expand the data points using LLM and utlise as well Data Science skills as possible

## Initial Idea:
### Take dataset from Kaggle about information on Airports, use LLM to get information on climate of the region, such as number of rainy days per year, number of foggy days, average temperature etc
<br>

### The combination of extra datapoints could allow some trends in the data to be linked to flight accidents, flight delays, flight cancellations.
### APIs of flight search may be used as well as LLM APIs 
#### It is important to demonstrate data cleaning, data gathering, data manipulation as well as the use of LLM and regression algorithms



## Dataset used
### flight_delay_data.csv : https://www.kaggle.com/datasets/aadharshviswanath/flight-data
### Airports-Only.csv : https://www.kaggle.com/datasets/thoudamyoihenba/airports
### World_Airports.csv : https://www.kaggle.com/datasets/mexwell/world-airports
### airlines.csv :  https://www.kaggle.com/code/fabiendaniel/predicting-flight-delays-tutorial/input?select=flights.csv
### airports.csv : https://www.kaggle.com/code/fabiendaniel/predicting-flight-delays-tutorial/input?select=flights.csv
### flights.csv : https://www.kaggle.com/code/fabiendaniel/predicting-flight-delays-tutorial/input?select=flights.csv
### Airplane_Crashes_and_Fatalities_Since_1908_t0_2023.csv : https://www.kaggle.com/datasets/nayansubedi1/airplane-crashes-and-fatalities-upto-2023
### daily_weather.parquet countries.csv cities.csv  : https://www.kaggle.com/datasets/guillemservera/global-daily-climate-data
#### World_Airports.csv, airports.csv and Airports-Only.csv should be matched and merged together

## Findings in this project
<br>
The project used datasets from multiple sources and analyses are conducted to determine which datasets can be used 
together to obtain airport information and location.
I have demonstrated data cleaning and filtering techniques and used LLMs for obtaining additional information.
In addition, innovative techniques such as calculating absolute distances between coordinates were used to confirm 
the locations are accurate. Anomalies are detected and removed.
<br>
I have discovered and confirmed that the location's weather data provided by Llama 3 is inaccurate and varies 
widely with ChatGPT 4 answers.


## Ideas for future development
<br>
The master dataset can be combined with flight data to understand if there are relationships between airport weather 
and the delays associated with a particular route.


## Useful material:
tutorial and example https://www.kaggle.com/code/fabiendaniel/predicting-flight-delays-tutorial