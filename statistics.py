import array
import matplotlib.pyplot as plt
from sys import argv
filename=argv[1]

with open(filename) as file:
	alllines=file.readlines()
values=[float(line.split('\t')[2].strip('\n')) for line in alllines[1:]]
a=array.array('f', values)
plt.hist(a, bins='auto')
plt.title("Histogram with 'auto' bins")

plt.show()