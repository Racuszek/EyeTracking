#!/usr/bin/env python3
"""
Takes 1 required argument 1 optional
required-file to get data  
optional-name of output files (out as default)

output: 
-to console coordinates of points [x,y] :
	left top corner 
	right top corner 
	left bottom corner 
	right bottom corner 
	middle at the beginning of file 
	middle at the end of file
-as *.txt file:
	as above
-*.png file 
	visualization of points
"""
from sys import argv
from matplotlib import pyplot as plt

SIDES_LENGHT_DISCARD = 0.5 # In percent how much of the focus time we discard 
TAIL_DISCARD = 0.00 # In percent how much of the file ending we discard
ELEMENT_COUNT = 6
PICTURE_SIZE = (1024,768)

def avrage (list):
	return sum( list )/len(list)


def isnumber(val):
	try:
		float(val)
		return True
	except ValueError as e:
		return False

#File neme missing
if argv.__len__() == 1:
	print("Point to calibration file plis :-(")
	exit()

#open calibration file seperated with tabs
try:
	with open(str(argv[1]),"r",encoding='ISO-8859-1') as file:
		data = [[int(line.split("\t")[1].strip()),int(line.split("\t")[2].strip())] for line in  file if line.split("\t").__len__() >= 3 and  line.split("\t")[0].strip().isdigit() and isnumber(line.split()[0]) ]
except FileNotFoundError as e:
	print("wrong file name :(")
	exit() 
#Calculate usefull things
lenght = data.__len__() 
lenght = lenght - lenght * TAIL_DISCARD
data = data[0:int(lenght)]
lenght = data.__len__() 
segmentLenght = lenght/ELEMENT_COUNT
elementLenght = (lenght/ELEMENT_COUNT) - (lenght/ELEMENT_COUNT)*SIDES_LENGHT_DISCARD
offset = int(((lenght/ELEMENT_COUNT)*SIDES_LENGHT_DISCARD)/2)

#get middle view data
middle = data[int(offset + 0*(segmentLenght)):int((offset+elementLenght)+(0*segmentLenght))]
middle2= data[int(offset + 5*(segmentLenght)):int((offset+elementLenght)+(5*segmentLenght))]

#calculate middle point
x1 = avrage(( tuple((value[0]) for value in middle)) )
y1 = avrage(( tuple((value[1]) for value in middle)) )
x2 = avrage(( tuple((value[0]) for value in middle2)))
y2 = avrage(( tuple((value[1]) for value in middle2)))
mid = (avrage( tuple( (x1,x2) )),avrage(	tuple( (y1,y2) )) )
#print(mid)

#remove value of middle from data to remove offset
data = [[line[0]-mid[0],line[1]-mid[1]] for line in data ]

leftTop		= data[int(offset + 1*(segmentLenght)):int((offset+elementLenght)+(1*segmentLenght))]
rightTop	= data[int(offset + 2*(segmentLenght)):int((offset+elementLenght)+(2*segmentLenght))]
leftBottom	= data[int(offset + 3*(segmentLenght)):int((offset+elementLenght)+(3*segmentLenght))]
rightBottom	= data[int(offset + 4*(segmentLenght)):int((offset+elementLenght)+(4*segmentLenght))]
middle 		= data[int(offset + 0*(segmentLenght)):int((offset+elementLenght)+(0*segmentLenght))]
middle2		= data[int(offset + 5*(segmentLenght)):int((offset+elementLenght)+(5*segmentLenght))]

leftTop		= [avrage(tuple((value[0] for value in leftTop ))), avrage(tuple((value[1] for value in leftTop ))) ] 
rightTop	= [avrage(tuple((value[0] for value in rightTop ))), avrage(tuple((value[1] for value in rightTop ))) ] 
leftBottom	= [avrage(tuple((value[0] for value in leftBottom ))), avrage(tuple((value[1] for value in leftBottom ))) ] 
rightBottom	= [avrage(tuple((value[0] for value in rightBottom ))), avrage(tuple((value[1] for value in rightBottom ))) ] 
middle 		= [avrage(tuple((value[0] for value in middle ))), avrage(tuple((value[1] for value in middle ))) ] 
middle2		= [avrage(tuple((value[0] for value in middle2 ))), avrage(tuple((value[1] for value in middle2 ))) ] 

ValueList = ([leftTop[0],leftBottom[0],rightBottom[0],rightTop[0],middle[0],middle2[0]],[leftTop[1],leftBottom[1],rightBottom[1],rightTop[1],middle[1],middle2[1]])

#output of files 

writeFileName =  "out" + argv[1] 
if argv.__len__() == 3:
	writeFileName = argv[2]
with open(writeFileName + ".dat","w") as file:
	for i in range (6):
		file.write("{}\t{}\n".format(ValueList[0][i],ValueList[1][i]))

print("{}\n{}\n{}\n{}".format(leftTop,rightTop,leftBottom,rightBottom))

plt.plot(ValueList[0],ValueList[1],"+")
plt.grid(True)
#plt.axis([-,320,-250,200])
plt.savefig(writeFileName + ".png")
