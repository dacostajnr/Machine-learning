from PIL import Image,ImageFilter
import random




def read_image(imagefile):
	#-----------------------------------------------------------
	# Returns an image Object
	#-----------------------------------------------------------
	image = Image.open(imagefile)
	return image 


def display_image(image_object):
	image_object.show()



def save_image(image_object,extension,name="image "+str(random.randint(0,100))):
	#-----------------------------------------------------------
	# Takes an {image object} and saves it
	#-----------------------------------------------------------
	image_object.save(name+"."+extension)



def convert_image(image_object):
	#-----------------------------------------------------------
	#Converts an {image_file} from {ext1} to {ext2}
	#-----------------------------------------------------------
	pass


def black_white_image(image_object):
	#-----------------------------------------------------------
	#Converts an {image_file} to black and white
	#-----------------------------------------------------------
	pass


def blur_image(image_object):
	#---------------------------------------------------------------
	# Blurs an image
	#---------------------------------------------------------------
	pass


def resize_image(image_object,w,h,extention="jpg"):
	image_object.thumbnail((w,h))
	save_image(image_object,extention)


def rotate_image():
	pass





def main():
	i = read_image("image.jpg")
	resize_image(i,100,100)
	#save_image(i,extension="png")


if __name__=="__main__":
	main()
	
