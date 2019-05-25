with open('zofia_jogurt.txt') as file:
	alllines=file.readlines()
output=open('zofia_jogurt_negativeoffset.txt', 'w+')
output.write(alllines[0])
for line in alllines[1:]:
	no=line.split('\t')[0]
	x=line.split('\t')[1]
	y=float(line.split('\t')[2].strip('\n'))-300.
	output.write(no+'\t'+x+'\t'+str(y)+'\n')
output.close()