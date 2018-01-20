from multiprocessing import Pool
import requests
import json
import time

def post_request(text):
        s = time.time()
	response = requests.post('http://localhost:5000', json={'post_text': text})
        #response = json.loads(response.text)
        return time.time()-s


p = Pool(5)
latencies = p.map(post_request, ['hello', 'battery'])
print(latencies)

