import numpy as np
import csv
import struct
import os

def read(filename):
	
	with open(filename, mode='rb') as f:
		filecontent=f.read()

	d=ord(filecontent[0])
	format=filecontent[1:1+d]
	print(format)
	w=struct.calcsize(format)
	print(w)
	currentIndex=1+d
	header=[]
	for i in range(d):
		l=ord(filecontent[currentIndex])
		header.append(str(filecontent[currentIndex+1:currentIndex+1+l]))
		currentIndex=currentIndex+1+l

	data=[]
	while(currentIndex+w<=len(filecontent)):
		#print len(filecontent[currentIndex:currentIndex+w])
		data.append(struct.unpack(format, filecontent[currentIndex:currentIndex+w]))
		currentIndex=currentIndex+w

	return header,data

if __name__ == '__main__':
	header,table=read('test.table')
	with open('out4.csv','wb') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		csvwriter.writerow(header)
		for row in table:
			csvwriter.writerow(row)
	#print header
	#for row in table:
	#	print row
	#save to CSV (with header)


