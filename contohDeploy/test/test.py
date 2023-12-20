import requests
url = "https://umkm-oyahvll4na-as.a.run.app"
# Make a POST request with parameter 'x'
ID_user = 44
resp = requests.post(url, json={"x": ID_user})

# Print the response
print(resp.json())