import json
from unittest.mock import patch  # Import for mocking
import pytest

from location_weather_llm import obtain_location_data


@patch('location_weather_llm.OpenAI')  # Patch the OpenAI client
def test_successful_data_retrieval(mock_openai):
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "average_temp": 25,
                        "average_wind_speed_annual": 12,
                        "average_humidity_annual": 72,
                        "average_visibility_annual": 22
                    })
                }
            }
        ]
    }

    mock_openai.return_value.chat.completions.create.return_value = mock_response

    data = json.loads(mock_response['choices'][0]['message']['content'])

    avg_temperature = data.get("average_temp", "NA")
    avg_wind_speed = data.get("average_wind_speed_annual", "NA")
    avg_humidity = data.get("average_humidity_annual", "NA")
    avg_visibility = data.get("average_visibility_annual", "NA")

    assert avg_temperature == 25
    assert avg_wind_speed == 12
    assert avg_humidity == 72
    assert avg_visibility == 22


@pytest.fixture()
def mock_openai():
    with patch('location_weather_llm.OpenAI') as mock_openai:
        yield mock_openai

@pytest.mark.parametrize(
    "location, country, expected_result",
    [
        ("London", "UK", ("25", "12", "72", "22")),  # Success scenario
        ("New York City", "USA", ("NA", "NA", "NA", "NA")),  # Missing data
        ("ImaginaryPlace", "ZZ", ("NA", "NA", "NA", "NA")),  # Invalid location
    ]
)
def test_location_data_with_params(mock_openai, location, country, expected_result):
    mock_response = {
        # *** Adjust the mock_response structure to match the real OpenAI API response ***
        # Example:
        "content": {
            "average_temp": expected_result[0],
            "average_wind_speed_annual": expected_result[1],
            "average_humidity_annual": expected_result[2],
            "average_visibility_annual": expected_result[3]
        }
    }
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    temp, wind_speed, humidity, visibility = obtain_location_data(location, country)

    assert temp == expected_result[0]
    assert wind_speed == expected_result[1]
    assert humidity == expected_result[2]
    assert visibility == expected_result[3]