import urllib.request
import json

# Define the API URL and parameters
base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
location = "Louisville,CO"
api_key = "9KPKUT3QZN76YU2UTBL2CUF4T"

# Construct the full URL with required parameters
url = (f"{base_url}{location}?unitGroup=us&contentType=csv"
       f"&include=hours&forecastBasisDate=2025-01-01&elements=datetimeEpoch,latitude,longitude,windspeedmean,winddir"
       f"&key={api_key}")

try:
    # Make the request
    response = urllib.request.urlopen(url)
    csv_data = response.read().decode('utf-8')
    
    # Print the retrieved CSV data
    print("Weather Data in CSV format:")
    print(csv_data)

except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} - {e.reason}")
    error_msg = e.read().decode()
    print("Error message:", error_msg)
except urllib.error.URLError as e:
    print(f"URLError: {e.reason}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")