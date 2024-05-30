from urllib import request

url = "https://www.chinanews.com.cn/cj/2024/03-28/10188704.shtml"

response = request.urlopen(url) #获取网页
print(response)
html = response.read() # 将网页的html读取下来
print(html)