# files and folders

import sys
import random
import os
import time

def changeDirectory(path):
	# changes the working directory to {path}
	pass

def readFile(filepath):
	# return file data
	pass


def writefile(filepath):
	pass
def writeZipFile(file,zipname):
	pass


def writeContactFile(filepath,name,number):
	file =open(filepath,"a")
	file.write("""
	BEGIN:VCARD
	VERSION:2.1
	N:;{};;;
	FN:{}
	TEL;CELL:{}
	END:VCARD""".format(name,name,number))

def renameFile(oldname,newname):
	pass


def deleteFile(file):
	pass


def getAllfiles(folderpath):
	pass


def readFileBytes(filepath):
	pass



import os
def isFile(path):
	"""
	Return true if path is a File
	""" 
	if not os.path.isdir(path):
		return True 
	return False


def listdir(path):
	"""
	Return the list of immediate children of folder
	"""
	return [i for i in os.listdir(path)]


def joinPaths(folder,file):
	"""
	Joins folder to filename
	"""
	return os.path.join(folder,file)


def getSize(path):
	
	return os.path.getsize(path)

def getFullSize(path):
	try:
		print(path)
		if isFile(path):
			return getSize(path)
		current = getSize(path)
		print("current size is {}".format(current))
		total_size=current+sum([getFullSize(joinPaths(path,inside)) for inside in listdir(path)])		
		print("================================================================================")
		print(total_size)
		print("=================================================================================")
		return total_size

	except:
		pass



def create_contacts():
	for i in range(0,10):
		contact=""
		networks=["027","057","020","050","024","054","055","026"]
		_list="0123456789"
		contact+=random.choice(networks)
		for j in range(0,7):
			contact+=random.choice(_list)
		print(contact)
		writeContactFile("new_contact.vcf","#"+str(i),contact) 


def main():
	create_contacts()
	



if __name__=="__main__":
	main()
