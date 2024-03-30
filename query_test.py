import http.client
from urllib.parse import quote
# 发送 GET 请求到服务器
def send_get_request(path, query_params=None):
    conn = http.client.HTTPConnection("localhost", 8000)
    url = path
    if query_params:
        url += "?" + "&".join([f"{key}={value}" for key, value in query_params.items()])
    conn.request("GET", url)
    response = conn.getresponse()
    print(f"Response Status: {response.status}")
    print(f"Response Body:\n{response.read().decode('utf-8')}")

# 测试 GET 请求
import json
print("Sending GET request...")
json_str=json.dumps({"smiles_list": ["C=C1COC1CC=CC","Cc1ccc2oc-2co1","C=C1C#CC(CC)=CO1","CCC1COCOC1C","C=C1CCOC(C)O1",
                                       "CCOC=COCOCC","CC1(C)CC2COC1C2","C=CCC(=C)OC(C)OC","CCOC(COC)OCC","C=CC1=CCC#COC1",
                                       "C=COC=CC(=C)COC","CC12CC13COCC3O2","CCC(C)=CCC(C)=O","COC12C#CCCC1C2","C#CC1(C)CC#CC1=O"]})
encoded_value = quote(json_str)
send_get_request("/", {"encoded_value":encoded_value})
print()