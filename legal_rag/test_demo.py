import requests

BASE = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"
body = {"sortDirection": "desc", "sortBy": "viewCount", "sortByViewCount": True,
        "pageNumber": 1, "pageSize": 3}

r = requests.post(f"{BASE}/all", json=body, timeout=30)
data = r.json()["data"]
print(data["total"])                      # tổng số văn bản
for item in data["items"]:
    print(item["id"], item["docNum"])

detail = requests.get(f"{BASE}/{data['items'][0]['id']}", timeout=30).json()["data"]
print(detail["title"])
print(detail["documentContent"]["content"][:300])   # HTML nội dung
