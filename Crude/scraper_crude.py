from bs4 import BeautifulSoup
import requests
import pandas


def datetime_to_float(d):
    return d.timestamp()


def Scrapdata(dataTable):
    Data = {}
    CrudePrice = []
    print(dataTable.find_all('td'))
    for td in dataTable.find_all('td'):
        # trs=tr.find_all('td')
        # print("tr",tr.string)
        # for td in tr.find_all('td'):
            # print("td ", td.string)
        try:
            CrudePrice.append(float(td.string))
        except:
            None
    print(CrudePrice)
    CrudePrice = CrudePrice[-200:]
    df = pandas.DataFrame(
        data={'Price': CrudePrice})
    df.to_csv("./CrudePrice.csv", sep=',', index=False)
    return Data


url = 'https://www.eia.gov/dnav/pet/hist/RWTCD.htm'
url_get = requests.get(url)
soup = BeautifulSoup(url_get.content, "html5lib")
# print(soup)
dataTable = soup.find('table', attrs={
                      "summary": "Cushing, OK WTI Spot Price FOB  (Dollars per Barrel)"})
# print(dataTable.find_all('tr'))
Data = Scrapdata(dataTable)
# JSON=json.dumps(Data)
# pickle.dump(JSON,fileObject)
# with open("output.txt", 'w') as file_handler:
#     file_handler.write(JSON)
