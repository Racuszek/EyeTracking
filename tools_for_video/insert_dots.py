#!/usr/bin/env python3.6
import cv2 
from glob import glob as glob #glob, glob? Glob!
sec_to_skip1 = 24*2
sec_to_skip2 = 24*2
def isnumber(val):
	try:
		float(val)
		return True
	except ValueError as e:
		return False


data =  glob(r"./data/*")
frames =  glob(r"./output/*")
print(data)
try:
	with open(data[0],"r",encoding='ISO-8859-1') as file:
		data1 = [[int(line.split('\t')[0].strip()),int(float(line.split("\t")[1].strip()))-200,int(float(line.split("\t")[2].strip()))-300] for line in  file if line.split("\t").__len__() >= 3 and  line.split("\t")[0].strip().isdigit() and isnumber(line.split()[0]) ]
except FileNotFoundError as e:
	print("wrong file name :(")
	exit() 
try:
	with open(data[1],"r",encoding='ISO-8859-1') as file:
		data2 = [[int(line.split('\t')[0].strip()),int(float(line.split("\t")[1].strip())),int(float(line.split("\t")[2].strip()))] for line in  file if line.split("\t").__len__() >= 3 and  line.split("\t")[0].strip().isdigit() and isnumber(line.split()[0]) ]
except FileNotFoundError as e:
	print("wrong file name :(")
	exit() 
data1 =data1[::-1]
data2 =data2[::-1]
for _ in range(sec_to_skip1):
	data1.append(None)
for _ in range(sec_to_skip2):
	data2.append(None)
frames.sort()
inti = 0
for frame in frames:
	pic = cv2.imread(frame)
	for _ in range(int(1000/24)):
		pop = data1.pop()
		if pop is not None:
			for x in range(-2,3):
				for y in range(-2,3):
					place = ((pop[2]-y if pop[2]-y<720 else 719) if pop[2]-y>0 else 0,(pop[1]-x if pop[1]-x<1280 else 1279) if pop[1]-x>0 else 0)
					pic.itemset((place[0],place[1],0),255)
					pic.itemset((place[0],place[1],1),0)
					pic.itemset((place[0],place[1],2),255)
		pop = data2.pop()
		if pop is not None:
			for x in range(-2,4):
				for y in range(-2,4):
					place = ((pop[2]-y if pop[2]-y<720 else 719) if pop[2]-y>0 else 0,(pop[1]-x if pop[1]-x<1280 else 1279) if pop[1]-x>0 else 0)
					pic.itemset((place[0],place[1],0),0)
					pic.itemset((place[0],place[1],1),100)
					pic.itemset((place[0],place[1],2),255)
	cv2.imshow("help me ", pic)
	cv2.waitKey(0)
	cv2.imwrite(frame.replace(r"output","output2"),pic)
	inti+=1
	if inti%50 == 0:
		print( str(inti) ,end="\t")

