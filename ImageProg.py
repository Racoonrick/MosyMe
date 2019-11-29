import sys
import os
import time
import math
import numpy as np
from PIL import Image
import colorsys

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

class ImageProg():
	def __init__(self):
		print("Image prog started")

	def CollageCalc(self):
		t0 = time.time()
		photos = 600
		im = Image.open('paris.jpg')

		width,height = im.size
		print("Width: ",width,"Height: ",height)
		aratio = width/height
		#Ex RGB has 3 colors 
		#set colors for different color schemes
		colors=3
		swid = math.floor(math.sqrt(aratio*photos))
		sheight = math.floor(math.sqrt(photos/aratio))
		print(swid,sheight)
		xpixs = math.floor(width/swid)
		ypixs = math.floor(height/sheight)
		self.x = (width/swid)
		self.y = (height/sheight)
		#Generates Filters for border and pixels
		#pover should be multiple of 2
		#power should be multiple of 2
		#filters are powX or none
		power = 0
		ftype = "none"
		pover = 0
		xxp,yxp = self.filter(xpixs,ypixs,power,ftype)
		xxpb,yxpb = self.filter(xpixs+pover*2,ypixs+pover*2,power,ftype)


		#pix1 = im.getpixel((1700,5000))
		#pix2 = im.getpixel((1700,4550))

		#converts image to floating point
		imc = im.convert(mode="RGB")
		data = np.array(imc)
		#print(data.shape)
		print(data[0,1],'\n\n\n\n')
		#print(data[65:72,55:57,:1],'\n')
		#print(data[0:2][0:2][:,1])
		val2 = [0,0,0]
		valh = []
		
		print(time.time()-t0)
		imgen = np.zeros((sheight,swid,colors))
		for y in range(0,sheight):
			for x in range(0,swid):
				#There are two filters
				#One for border
				#only used if pover > 0
				#pover filters over the desired area
				if pover > 0:
					if x == 0 or x == swid:
						xfilt = xxp
						xover = 0
					else:
						xfilt = xxpb
						xover = pover
					if y == 0 or y == swid:
						yfilt = yxp
						yover = 0
					else:
						yfilt = yxpb
						yover = pover
				else:
					xfilt = xxp
					xover = 0
					yfilt = yxp
					yover = 0
				print(x,y)
				xa=(x)*xpixs-xover
				xb=(x+1)*xpixs+xover
				ya=(y)*ypixs-yover
				yb=(y+1)*ypixs+yover
				for c in range(0,colors):
					#print(data[ya:yb,xa:xb,(c)],"\n\n\n",data[ya:yb,xa:xb,(c)].shape,xxp.shape,yxp.shape)
					val1 = np.dot([data[ya:yb,xa:xb,c]],xfilt)
					valh = np.dot(val1,yfilt)
					valr = valh.round()
					val2[c] = (valr/(xpixs+xover*2)/(ypixs+yover*2))
					#input("Press enter to continue...")
				imgen[y][x] = val2
		imS = Image.fromarray(imgen.astype('uint8'),'RGB')
		#imS.show()
		imS.save("scaledparis"+"pover"+str(pover)+"power"+str(power)+ftype+".jpg")
		t1 = time.time()
		print(t1-t0)
		return imgen
	def CollageGen(self):
		t0=time.time()
		imref = Image.open('scaledparispover0power0none.jpg')
		cdata = np.array(imref.convert(mode="RGB"))
		sourceImage = 'refpic4.jpg'
		#Color corrected data averaged for each
		cCorrectData = np.array(imref.convert(mode="RGB"))
		yref,xref,zref = cdata.shape
		#Finds the maximum we can deviate
		#for each reference color
		maxCorrectData = np.zeros((yref,xref))
		maxCorrectData = np.array(maxCorrectData)
		maxCorrectData = np.max(np.abs(cCorrectData-127.5),axis=2)
		maxCorrectData = 127.5 - maxCorrectData
		print(yref,xref,zref)

		imsource = Image.open(sourceImage)
		xrefsize,yrefsize = imsource.size
		#converts image to floating point
		sdata = np.array(imsource.convert(mode="RGB"))
		print(sdata.shape)
		yoffset = 0
		xoffset = 0
		imagecorrected = np.zeros((yrefsize*(yref-yoffset),xrefsize*(xref-xoffset),3),dtype=np.uint8)
		print(imagecorrected.shape)
		for y in range(0,(yref-yoffset)): 
			print(y)
			for x in range(0,(xref-xoffset)):
				#print(x,y)
				#imagecorrected[(y)*yrefsize:(y+1)*yrefsize,(x)*xrefsize:(1+x)*xrefsize,:] = ImageColorCorrect(sourceImage,cCorrectData[y,x],maxCorrectData[y,x])
				imagecorrected[(y)*yrefsize:(y+1)*yrefsize,(x)*xrefsize:(1+x)*xrefsize,:] = self.ImageBlend(imsource,cCorrectData[y,x])
		imOut = Image.fromarray(imagecorrected.astype('uint8'),'RGB')
		imOut.save("collage.jpg")
		
		imOut.show()

		#imOut.close()
		t1 = time.time()
		print(t1-t0)

	def OpenImageAsArray(self,FilePath):
		ImageSource = Image.open(FilePath)
		ImageArray = np.array(ImageSource.convert(mode="RGB"))
		return ImageArray

	def OpenImage(self,FilePath):
		ImageSource = Image.open(FilePath)
		return ImageSource

	def SaveImageFromArray(self,ImgArray,FileName):
		ImgFromArray = Image.fromarray(ImgArray.astype('uint8'),'RGB')
		ImgFromArray.save(FileName)

	def ImageBlend(self,ImageSource,cCorrectData):
		size = ImageSource.size
		layer = Image.new('RGB', size, cCorrectData) # "hue" selection is done by choosing a color...
		output = np.array(Image.blend(ImageSource, layer, 0.75))
		return output

	def ImageColorCorrect(self,sourceImage,cCorrectData,maxCorrectData):
		colors = 3
		
		imsource = Image.open(sourceImage)
		#converts image to floating point
		sdata = np.array(imsource.convert(mode="RGB"))
		imsource.close()
		y,x,z = sdata.shape
		odata = np.ones((y,x,z))
		meandataarray = np.zeros((y,x))
		correcteddataarray = np.zeros((y,x,z))
		scalearray = np.zeros((y,x,z))
		meandataarray = np.array(meandataarray)
		correcteddataarray = np.array(correcteddataarray)
		scalearray = np.array(scalearray)
		meanc = [0,0,0]
		# for c in range(0,colors):
		# 	meanc[c] = np.average(sdata[:,:,c])
		sdata = sdata/255
		meandataarray[:,:] = np.sum(sdata,axis=2)/3
			
		for c in range(0,3):
			correcteddataarray[:,:,c] = sdata[:,:,c]-meandataarray
			correcteddataarray[:,:,c] = correcteddataarray[:,:,c]*maxCorrectData+cCorrectData[c]
		
		# for c in range(0,colors):
		# 	odata[:,:,c] = (sdata[:,:,c]-127.5)/meandata[:,:,c]*cCorrectData[c]
		# 	odata[:,:,c] = odata[:,:,c]+127.5
		# 	print(sdata[124,150,c],cCorrectData[c],meanc[c],np.max(odata))
		nphold = np.array(correcteddataarray)
		#np.savetxt('tf.csv',odata[:,:,1],delimiter=",")
		#imOut = Image.fromarray(nphold.astype('uint8'),'RGB')
		#test = nphold.astype(int)
		return nphold.astype(np.uint8)


	def filter(self,x,y,exp,ftype):
		if ftype == "powX":
			#Generates x,y range based on input
			xw = np.array(range(1,x+1))
			yw = np.array(range(1,y+1))
			#Centers range around 0
			xws = np.subtract(xw,x/2+0.5)
			yws = np.subtract(yw,y/2+0.5)
			#Raises to exp and normalizes
			xxp = -1*xws**exp/xws[0]**exp+1
			yxp = -1*yws**exp/yws[0]**exp+1
			xxp = xxp/np.mean(xxp)
			yxp = yxp/np.mean(yxp)
			#xdp = np.dot(ar,xxp)
			#ydp = np.dot(xdp,yxp)
		elif ftype == "none":
			xxp = np.ones(x)
			yxp = np.ones(y)
		return xxp,yxp

if __name__ == '__main__':
	test=ImageProg()
	test.CollageCalc()
	#filter()
	#ImageColorCorrect(30,18,'refpic3.jpg')
	test.CollageGen()