import requests
import json

url = "http://127.0.0.1:5000"

# Integer value to send
input_integer = 42  # Replace this with your actual integer value

# Create a dictionary with the integer value
data = {'value': input_integer}

# Send the POST request with JSON data
resp = requests.post(url, json=data)

# Print the response
print(resp.json())