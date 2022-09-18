# import BaseHTTPServer
import json
from ocr import OCRNeuralNetwork
import numpy as np

HOST_NAME = 'localhost'
PORT_NUMBER = 8000
HIDDEN_NODE_COUNT = 15

# Load text  data samples and labels into matrix
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter=',')
print(np.shape(data_matrix))
data_labels = np.loadtxt(open('dataLabels.csv', 'rb'))
print(np.shape(data_labels))

print("====================")

# Convert from numpy ndarrays to python lists
data_matrix = data_matrix.tolist()
print(len(data_matrix))
print(len(data_labels))

# If a neural network file does not exist, train it using all 5000 existing data samples.
# Based on data collected from neural_network_design.py, 15 is the optimal number
# for hidden nodes
# 通过类名实例化的话，直接调用的是init发方法返回一个实例对象,，先利用5000个数据简单训练一个神经网络
nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, list(range(5000)));

# class JSONHandler(BaseHTTPServer.BaseHTTPRequestHandler):
#     def do_POST(s):
#         response_code = 200
#         response = ""
#         varLen = int(s.headers.get('Content-Length'))
#         content = s.rfile.read(varLen);
#         payload = json.loads(content);
#
#         if payload.get('train'):
#             nn.train(payload['trainArray'])
#             nn.save()
#         elif payload.get('predict'):
#             try:
#                 response = {"type":"test", "result":nn.predict(str(payload['image']))}
#             except:
#                 response_code = 500
#         else:
#             response_code = 400
#
#         s.send_response(response_code)
#         s.send_header("Content-type", "application/json")
#         s.send_header("Access-Control-Allow-Origin", "*")
#         s.end_headers()
#         if response:
#             s.wfile.write(json.dumps(response))
#         return
#
# if __name__ == '__main__':
#     server_class = BaseHTTPServer.HTTPServer;
#     httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)
#
#     try:
#         httpd.serve_forever()
#     except KeyboardInterrupt:
#         pass
#     else:
#         print "Unexpected server exception occurred."
#     finally:
#         httpd.server_close()
