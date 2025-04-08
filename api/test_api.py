import requests # type: ignore

# Get predictions for all stocks
response = requests.get('http://localhost:5000/predict')
print(response.json())

# Get prediction for specific stock (e.g., AAPL)
# response = requests.get('http://localhost:5000/predict/AAPL')
# print(response.json())
