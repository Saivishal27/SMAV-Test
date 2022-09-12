import requests
import json

url = "http://localhost:8081/gateway"
packet = {'data_file_name': '/datadrive2/ml_model_deployment/Vishal_test/test_data.csv',
        'request_type'  :'train',
        'use_cols'      :['CGIABVE','FRT','SPD','BHPCOR','BDENDPRS','CMPCYL3VIB','CMPCYL2VIB','CMPCYL4VIB','CGIBLWE'],
        'labels'        :['FRT','BHPCOR'],
        'window_size'   :1,
        'stride'        :1
         }
response = requests.post(url, json=packet)
print (response.text)
response = json.loads(response.text)


packet = {'data_file_name': '/datadrive2/ml_model_deployment/Vishal_test/test_data.csv',
        'request_type':'test',
        'model_type' : 'ML',
        'metrics' : ["mean_absolute_error"]
         }
response = requests.post(url, json=packet)
print (response.text)
response = json.loads(response.text)
