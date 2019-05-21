#!/usr/bin/env python3
import cv2 
from copy import deepcopy as cp
def organize(doublelist1,doublelist2):
	""" 
	returns values wiet zmoler y value and if there was switch. 
	"""
	if doublelist1[1]>doublelist2[1]:
		return True, cp(doublelist2),cp(doublelist1)
	else:
		return False, cp(doublelist1),cp(doublelist2)


def generate_matrix(cordList):
	"""
	takes python list as a cordynates:
		left top corner 
		right top corner 
		left bottom corner 
		right bottom corner 
		middle at the beginning of file 
		middle at the end of file
	returns:
		list of transforation matrixies to achive imput rectangle of:
			width:	1024
			hight:	768
	"""
	mid = [(cordList[4][0]+cordList[5][0])/2,(cordList[4][1]+cordList[5][1])/2]
	boll,lower,higher = organize( cordList[0],cordList[1])
	if mid[1]>higher[1]:
		print("i aint dealing with that shiet")
		return "fuk me thats bad"
	for i in [i/10 for i in range(90)]:
		mat = cv2.getRotationMatrix2D((mid[0],mid[1]),i,1)
		print(mat)