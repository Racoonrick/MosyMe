import sys
import os
import time
import math
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw
import colorsys

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

class ImageProg():
	def __init__(self):
		print("Image prog started")

	def CollageCalc(self):
		t0 = time.time()
		photos = 600
		im = Image.open('parisFiltered.jpg')

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
		pover = -8
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
					if x==0 and y==28:
						print('ya: '+str(ya))
						print('yb: '+ str(yb))
						print('xa: '+str(xa))
						print('yxp: '+ str(yxp.shape))
						print('yfilt: '+str(yfilt.shape))
						print('data: '+str(data.shape))
						print('swid: '+str(swid))
						print('sheight: '+str(sheight))
						print(val1)
					#valh = np.dot(val1,yfilt)
					#valr = valh.round()
					#val2[c] = (valr/(xpixs+xover*2)/(ypixs+yover*2))
					valr = np.median(data[ya:yb,xa:xb,c])
					#valr = np.argmax(valh)
					val2[c] = valr
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
		imref = Image.open('gimpParis.jpg')
		#Open the mask as the filter
		immask = Image.open('gimpParisThreshold.jpg')
		cdata = np.array(imref.convert(mode="RGB"))
		sourceImage = 'refpic.jpeg'
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
		layer = Image.new('RGB', size, (cCorrectData[0],cCorrectData[1],cCorrectData[2])) # "hue" selection is done by choosing a color...
		output = np.array(Image.blend(ImageSource, layer, 0.75))
		return output

	def NewImageBlend(self,ImageSource,cCorrectData):
		size = ImageSource.size
		layer = Image.new('RGB', size, (cCorrectData[0],cCorrectData[1],cCorrectData[2])) # "hue" selection is done by choosing a color...
		output = Image.blend(ImageSource, layer, 0.75)
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

	def countWhitePixels(self,thresholdPhotoPath):
		#Initialize counting function
		#Also Generates an array which can be used
		#	as the reference for where each photo goes
		#	based on the threshold generated image
		#
		#Example thresholdPhotoPath: gimpParisThreshold.jpg
		#	
		#Only coordinates which are white will be added to the
		#	photo coordinates array
		#
		#The first photo will correspond to position 0 in the 
		#	photo coordinates array, which is [0,29] or x=0
		#	y=29, so on and so forth.
		#
		countcolor=0
		counttotal=0
		photoCoordinates=[]
		im = Image.open(thresholdPhotoPath)
		width,height = im.size
		imc = im.convert(mode="RGB")
		data = np.array(imc)
		for x in range(0,width):
			for y in range(0,height):
				counttotal=counttotal+1
				avgcolor = np.average(data[y,x,0:3])
				if avgcolor>200:
					# If reference is white add it to list
					# possible photo coordinates
					countcolor=countcolor+1
					photoCoordinates.append([x,y])
		return countcolor,counttotal,photoCoordinates

	def imageCrop(self,im):
		# Currently unused
		#im=Image.open('../PhotoBooth/imFromCam/cam0003.jpg')
		#im.show()
		width,height=im.size
		left = (width-height)/2
		right = height+left
		im1=im.crop((left,0,right,height))
		return im1

	def imageOverlay(self,mainpath,borderpath,colorcorrectdata,coord):
		mainphoto = Image.open(mainpath)
		borderphoto = Image.open(borderpath)
		#Sets correct color information
		colors=colorcorrectdata[coord[1],coord[0]]
		#Creates White background image
		background = Image.new('RGBA', (2732, 1868), (255, 255, 255, 255))
		# Pastes mainphoto on white background
		background.paste(mainphoto,(70,70))
		# Overlays borderphoto on top of the mainphoto
		#compositephoto=Image.alpha_composite(background,borderphoto)
		compositephoto=background
		# Resizes the photo maintaining aspect ratio but fitting
		# within 600 pixels wide
		width,height = compositephoto.size
		dims=(600,math.floor(height/width*600))
		compositephoto=compositephoto.resize(dims)
		# Generates color corrected image for board
		# This requires coord to be given as an array
		# 	with 3 rows of data
		correctedphoto=self.NewImageBlend(mainphoto,colors)
		# Crops image to a square
		correctedphoto=self.imageCrop(correctedphoto)
		# Resizes corrected image
		correctedphoto=correctedphoto.resize((560,560))
		# Generates and places text over image
		strx=str(coord[0])
		stry=str(coord[1])
		strcoord='c'+strx+'r'+stry
		font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf", 35)
		draw=ImageDraw.Draw(correctedphoto)
		draw.text((420, 500),strcoord,(255,255,255),font=font)

		return compositephoto, correctedphoto
		

	def MakePrint(self,photos,border,colorcorrectdata,coord):
		# Loads main photo and places border over top
		# Returns the composite and the original photo
		imageborder=Image.open('/home/ricky/Desktop/PhotoBooth/ImagesGUI/JCbday2.png')
		composite1,corrected1=self.imageOverlay(photos[0],border,colorcorrectdata,coord[0])
		composite2,corrected2=self.imageOverlay(photos[1],border,colorcorrectdata,coord[1])
		composite3,corrected3=self.imageOverlay(photos[2],border,colorcorrectdata,coord[2])
		# Generates white background strip for print photo
		print(composite1.size)
		backgroundstrip=Image.new('RGBA',(1200,1800),(255,255,255,255))
		# Places the resized and corrected photos onto strip
		backgroundstrip.paste(composite1,(0,65))
		backgroundstrip.paste(composite2,(0,505))
		backgroundstrip.paste(composite3,(0,970))
		# backgroundstrip.paste(corrected1,(620,30))
		# backgroundstrip.paste(corrected2,(620,620))
		# backgroundstrip.paste(corrected3,(620,1210))
		#Added for JCs bday
		backgroundstrip.paste(composite1,(600,65))
		backgroundstrip.paste(composite2,(600,505))
		backgroundstrip.paste(composite3,(600,970))
		backgroundstripf=Image.alpha_composite(backgroundstrip,imageborder)
		#backgroundstrip.paste(imageborder,(0,0))
		#backgroundstrip.paste(imageborder,(600,0))
		return backgroundstripf


if __name__ == '__main__':
	test=ImageProg()
	#print(test.countWhitePixels())
	#test.CollageGen()
	#test.imageCrop()
	countcolor,counttotal,photoCoordinates=test.countWhitePixels('gimpParisThreshold.jpg')
	imref = Image.open('gimpParis.jpg')
	cdata = np.array(imref.convert(mode="RGB"))
	photos=["../PhotoBooth/imFromCam/cam0001.jpg",
			"../PhotoBooth/imFromCam/cam0002.jpg",
			"../PhotoBooth/imFromCam/cam0003.jpg"]
	borderpath="../PhotoBooth/ImagesGUI/ImageBorderBasic.png"
	photoout=test.MakePrint(photos,borderpath,cdata,photoCoordinates[355:358])
	photoout.save('PrintPhoto.png')