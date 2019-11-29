import sys
import os
import time
import math
import shutil
import numpy as np
from PIL import Image
import colorsys
from ImageProg import ImageProg

class ImageHandler():
	def __init__(self):
		print("ImageHanlder Initialized")
		self.InPath = "C:\\Users\\rwahl\\Documents\\Work\\MosyMe\\ProcImage\\InBoundImg"
		self.OutPath = "C:\\Users\\rwahl\\Documents\\Work\\MosyMe\\ProcImage\\OutBoundImg"
		self.ProcessedPath = "C:\\Users\\rwahl\\Documents\\Work\\MosyMe\\ProcImage\\ProcessedImg"
		self.ImgCount = 1
		self.ip = ImageProg()

	def CheckFiles(self):
		FolderEmpty = True
		CheckTime = 2
		while FolderEmpty:
			time.sleep(CheckTime)
			if not os.listdir(self.InPath) == []:
				print("There's stuff in there")
				FilesList = [f for f in os.listdir(self.InPath) if os.path.isfile(os.path.join(self.InPath, f))]
				FolderEmpty=False
				self.ProcessFiles(FilesList)

	def ProcessFiles(self,FilesList):
		for f in FilesList:
			TestColor = (255,255,0)
			print(TestColor)
			ImIn = self.ip.OpenImage(self.InPath+"\\"+f)
			ImgCorrected = self.ip.ImageBlend(ImIn,TestColor)
			self.ip.SaveImageFromArray(ImgCorrected,self.ProcessedPath+"\\"+f[:-4]+str(self.ImgCount)+f[-4:])
			#shutil.copy2(self.InPath+"\\"+f,self.ProcessedPath+"\\"+f[:-4]+str(self.ImgCount)+f[-4:])
			shutil.move(self.InPath+"\\"+f,self.OutPath+"\\"+f[:-4]+str(self.ImgCount)+f[-4:])
			print(f)



if __name__ == '__main__':
	ImgH = ImageHandler()
	ImgH.CheckFiles()
	teststring="farttymc.jpg"
	print(teststring[-4:])
	print(teststring[:-4])