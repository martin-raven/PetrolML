from bs4 import BeautifulSoup
import requests
import json
import pickle
import numpy as np
from datetime import datetime
import csv
import pandas
def datetime_to_float(d):
    return d.timestamp()
def Scrapdata(dataTable):
	Data={}
	dataValue={}
	dates=[]
	DelhiPrice=[]
	tr_count=0
	for tr in dataTable.find_all('tr'):
		# trs=tr.find_all('td')
		# print("tr",tr)
		for td in tr.find_all('td'):
			# print("td",td)
			span_count=0
			for span in td.find_all('span'):
				# print(span_count,span)
				if span_count%5==0:
					dateObj=datetime.strptime(span.string, '%B %d, %Y')
					dataValue["date"]=span.string
					dataValue["dateFloat"]=datetime_to_float(dateObj)
					dates.append(dataValue["dateFloat"])
				elif span_count==1:
					dataValue["Delhi"]=span.string
					DelhiPrice.append(float(span.string))
				elif span_count==2:
					dataValue["Kolkata"]=span.string
				elif span_count==3:
					dataValue["Mumbai"]=span.string
				elif span_count==4:
					dataValue["Chennai"]=span.string
				# print(span_count,dataValue)
				span_count+=1
				if(span_count==5):
					span_count=0
					tr_count+=1
					Data[tr_count]=dataValue
					dataValue={}
					if(Data[tr_count]["date"]=='January 01, 2018'):
						Data["Totaldata"]=tr_count
						df = pandas.DataFrame(data={'Timestamp': dates, 'Weighted_Price': DelhiPrice})
						df.to_csv("./DelhiPrice.csv", sep=',',index=False)
						return Data
# file_Name = "DataSet"
# # open the file for writing
# fileObject = open(file_Name,'wb')
url = 'https://www.iocl.com/Product_PreviousPrice/PetrolPreviousPriceDynamic.aspx'
url_get = requests.get(url)
soup = BeautifulSoup(url_get.content,"html5lib")
dataTable=soup.find('table',attrs={"class":"wrapper-table"})
# print(dataTable.find_all('tr'))
Data=Scrapdata(dataTable)
# JSON=json.dumps(Data)
# pickle.dump(JSON,fileObject)
# with open("output.txt", 'w') as file_handler:
#     file_handler.write(JSON)