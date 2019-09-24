from PIL import Image 
from os.path import basename

def image_resize(path_img):
	#basewidth = 300
	img = Image.open(path_img)
	#wpercent = (basewidth/float(img.size[0]))
	#hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((100,128), Image.ANTIALIAS)
	img_out_path = basename(path_img) + "resized" + ".jpg"
	print(img_out_path)
	img.save(img_out_path)



mask_0 = "/Users/liujin/Desktop/mask_0.jpeg"
mask_1 = "/Users/liujin/Desktop/mask_2.jpeg"

if __name__ == "__main__":
	image_resize(mask_0)
	image_resize(mask_1)