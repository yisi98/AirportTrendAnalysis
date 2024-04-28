import json
from unittest.mock import patch  # Import for mocking

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