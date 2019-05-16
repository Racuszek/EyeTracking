import numpy as np
from sys import argv
import matplotlib.pyplot as plt

try:
	file=open(argv[1])
except:
	print('Error occured while opening file: '+argv[1])

plt.axis([700, 1400, 2100, 2500])

for i in range(20):
	file.readline()
	# I seriously don't know how to do it any better.

xlist=[]
ylist=[]

for i in range(25000):
	line=file.readline().split()
	# print(line)
	x=int(line[1])
	# xlist.append(x)
	y=int(line[2])
	# ylist.append(y)
	# x=np.random.randint(1410, 2160)
	# y=np.random.randint(1090, 1900)
	plt.scatter(x, y, s=10)	
	plt.pause(0.00001)
# print(xlist)
testxlist=[1080, 1090, 1085]
testylist=[2340, 2320, 2315]
# print(ylist)
plt.grid(True)

# plt.savefig('figure1.png')
# plt.show()

# print(file.readline().split()[1])
# print(file.readline().split()[2])