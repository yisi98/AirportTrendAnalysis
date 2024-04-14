"""
This file will use LLM to obtain the location's weather information
It will include prompting LLM API and formatting the response in a way to interpret into airport dataframe
"""
import json

import numpy as np
import pandas as pd
from chronological import read_prompt, cleaned_completion
from loguru import logger
from openai import OpenAI


def check_not_present_or_unsure(string):
    if "not present" in string:
        return "answer not present"
    elif "Unsure" in string or "unsure" in string:
        return "Unsure"
    return string


def obtain_location_data(location: str, country: str):
    # you call the Chronology functions, awaiting the ones that are marked await

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON. If you cannot find "
                                          "the value, put 'NA' and do not make up your answers. Example output should "
                                          "look like:: average_temp : 25, average_wind_speed_annual: 12, "
                                          "average_humidity_annual:  72, average_visibility_annual: 22,"},
            {"role": "user", "content": f"What is the average temperature throughout the year and give just a value "
                                        f"in Celcius, strictly not a range.  What is the average wind speed of that "
                                        f"city, provide one value in knots. What is the average percentage of "
                                        f"humidity in that city in a year, just provide the value without percentage "
                                        f"signs. What is the average visibility in that city in the year, "
                                        f"in km? Provide answer without additional text and strictly only the JSON "
                                        f"format below"}
        ]
    )

    data = json.loads(response.choices[0].message.content)
    avg_temperature = data['average_temp']
    avg_wind_speed = data['average_wind_speed_annual']
    avg_humidity = data['average_humidity_annual']
    avg_visibility = data['average_visibility_annual']
    return avg_temperature, avg_wind_speed, avg_humidity, avg_visibility


def run_llm_queries(df):
    # function which uses GPT to obtain datapoints such as weather
    # create new columns
    df["avg_temperature"] = np.nan
    df["avg_wind_speed"] = np.nan
    df["avg_humidity"] = np.nan
    df["avg_visibility"] = np.nan
    # run the queries
    for index, row in df.iterrows():
        avg_temperature, avg_wind_speed, avg_humidity, avg_visibility = obtain_location_data(row["original_city"],
                                                                                             row["country"])
        df.loc[index, "avg_temperature"] = avg_temperature
        df.loc[index, "avg_wind_speed"] = avg_wind_speed
        df.loc[index, "avg_humidity"] = avg_humidity
        df.loc[index, "avg_visibility"] = avg_visibility
        logger.info(f"country: {row['country']}, city: {row['original_city']}, avg_temperature: {avg_temperature}, "
                    f"avg_wind_speed: {avg_wind_speed}, avg_humidity: {avg_humidity}, avg_visibility: {avg_visibility}")

    return df


def main():
    # load the dataset
    airport_df = pd.read_csv("../data/airport_dataset_cleaned.csv")
    output_df = run_llm_queries(airport_df)
    output_df.to_csv("../data/airport_dataset_with_llm_weather.csv")


if __name__ == "__main__":
    main()
