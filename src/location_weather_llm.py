"""
This file will use LLM to obtain the location's weather information
It will include prompting LLM API and formatting the response in a way to interpret into airport dataframe
"""

import json
import os
import pandas as pd
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


def obtain_location_data(location: str, country: str):
    # you call the Chronology functions, awaiting the ones that are marked await

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON. If you cannot find "
                "the value, put 'NA' and do not make up your answers. Example output should "
                "look like:: average_temp : 25, average_wind_speed_annual: 12, "
                "average_humidity_annual:  72, average_visibility_annual: 22,",
            },
            {
                "role": "user",
                "content": f"What is the average temperature in {location} {country} throughout the year "
                f"and give just a value"
                f"in Celcius, strictly not a range.  What is the average wind speed of {location} {country}"
                f", provide one value in knots. What is the average percentage of "
                f"humidity in {location} {country} in a year, just provide the value without "
                f"percentage"
                f"signs. What is the average visibility in {location} {country} in the year, "
                f"in km? Provide answer without additional text and strictly only the JSON "
                f"format below",
            },
        ],
    )
    try:
        data = json.loads(response.choices[0].message.content)
        avg_temperature = data.get("average_temp", "NA")
        avg_wind_speed = data.get("average_wind_speed_annual", "NA")
        avg_humidity = data.get("average_humidity_annual", "NA")
        avg_visibility = data.get("average_visibility_annual", "NA")
        return (
            str(avg_temperature),
            str(avg_wind_speed),
            str(avg_humidity),
            str(avg_visibility),
        )
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return e, None, None, None


def run_llm_queries(df):
    checkpoint_folder = "../data/saved_checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)

    checkpoint_size = 100
    total_rows = len(df)
    all_dfs = (
        []
    )  # List to store dataframes from each checkpoint for final concatenation

    # Wrap range with tqdm for a progress bar
    for start in tqdm(
        range(0, total_rows, checkpoint_size), desc="Processing checkpoints"
    ):
        end = min(start + checkpoint_size, total_rows)
        checkpoint_path = (
            f"{checkpoint_folder}/checkpoint_{start // checkpoint_size}.csv"
        )

        if os.path.exists(checkpoint_path):
            logger.info(
                f"Skipping processing for {checkpoint_path} as it already exists."
            )
            checkpoint_df = pd.read_csv(checkpoint_path)
        else:
            checkpoint_df = df[start:end].copy()
            for index, row in tqdm(
                checkpoint_df.iterrows(),
                total=len(checkpoint_df),
                desc=f"Checkpoint {start // checkpoint_size}",
            ):
                avg_temperature, avg_wind_speed, avg_humidity, avg_visibility = (
                    obtain_location_data(row["original_city"], row["country"])
                )
                checkpoint_df.at[index, "avg_temperature"] = avg_temperature
                checkpoint_df.at[index, "avg_wind_speed"] = avg_wind_speed
                checkpoint_df.at[index, "avg_humidity"] = avg_humidity
                checkpoint_df.at[index, "avg_visibility"] = avg_visibility
                logger.info(
                    f"Processed {row['country']}, {row['original_city']} - Checkpoint {start // checkpoint_size} - {avg_temperature}"
                )

            checkpoint_df.to_csv(checkpoint_path, index=False)
            logger.info(f"Saved checkpoint at {checkpoint_path}")

        all_dfs.append(checkpoint_df)

    # Concatenate all dataframes to form the final dataframe
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df


def main():
    # load the dataset
    airport_df = pd.read_csv("../data/airport_dataset_cleaned.csv")
    output_df = run_llm_queries(airport_df)
    output_df.to_csv("../data/airport_dataset_with_llm_weather.csv")


if __name__ == "__main__":
    main()
