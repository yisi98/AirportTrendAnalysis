import json
import os

import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

url = "https://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}


def obtain_location_data_ollama(location: str, country: str):
    prompt_message = f"What is the average temperature in {location} {country} throughout the year and give just a " \
                     f"value in Celcius, strictly not a range.  What is the average wind speed of {location} " \
                     f"{country} provide one value in knots. What is the average percentage of humidity in {location} {country}" \
                     f"in a year, just provide the value without percentage signs. What is the average visibility in {location}" \
                     f"{country} in the year, in km? Provide answer without additional text and strictly only the JSON " \
                     f"format below " \
                     f"If you cannot find " \
                     f"the value, put 'NA' and do not make up your answers. Example output should " \
                     f"look like:: average_temp : 25, average_wind_speed_annual: 12, " \
                     f"average_humidity_annual:  72, average_visibility_annual: 22,"

    response = ollama.generate(model='llama3', prompt=prompt_message)
    try:
        data = json.loads(response["response"])
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


def run_ollmama_queries(df):
    checkpoint_folder = "../data/ollama_saved_checkpoints"
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
                    obtain_location_data_ollama(row["original_city"], row["country"])
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
    airport_df = pd.read_csv("../data/airport_dataset_cleaned.csv")
    logger.info("Starting")
    output_df = run_ollmama_queries(airport_df)
    logger.info("Finished")
    output_df.to_csv("../data/airport_dataset_with_ollama_weather.csv")


if __name__ == "__main__":
    main()
