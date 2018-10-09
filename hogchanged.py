from PIL import Image
import numpy as np
import csv

cell = [1, 1] # Number of pixels of width x height for each cell
incr = [1,1] # How much to increment when moving to next gradient
num_of_bins = 5 # Number of bins to sort gradients into (Divided between 0 and 180 degrees)
image_size = (4,5) # Tuple size of image

image_array = np.array([[125, 250, 200, 200], [250, 100, 13, 12], [200, 200, 178, 206], [200, 78, 38, 150], [100, 100, 100, 100]], dtype=np.uint8)

#uses a [-1 0 1 kernel]
def create_grad_array(image_array):
	image_array = Image.fromarray(image_array)
	if not image_array.size == image_size:
		print("RESHAPING ARRAY SIZE")
		image_array = image_array.resize(image_size, resample=Image.BICUBIC)
	
	image_array = np.asarray(image_array,dtype=float)

	# gamma correction
	#image_array = (image_array)**2.5

	# local contrast normalisation
	image_array = (image_array-np.mean(image_array))/np.std(image_array)

	max_w = 6
	max_h = 7
	
	#pad image 
	image_array = np.lib.pad(image_array, 1, 'constant', constant_values=(0))

	grad = np.zeros([max_h, max_w])
	mag = np.zeros([max_h, max_w])
	for h,row in enumerate(image_array):
		for w, val in enumerate(row):
			if h-1>=0 and w-1>=0 and h+1<max_h and w+1<max_w:
				dy = image_array[h+1][w]-image_array[h-1][w]
				dx = row[w+1]-row[w-1]+0.0001
				grad[h][w] = np.arctan(dy/dx)*(180/np.pi)
				if grad[h][w]<0:
					grad[h][w] += 180
				mag[h][w] = np.sqrt(dy*dy+dx*dx)
	
    

	new_grad = []
	new_mag = []
	row_length = len(grad[0])
	for i in range(1, len(grad)-1):
		new_grad.append(grad[i][1:row_length-1])

	for i in range(1, len(mag)-1):
		new_mag.append(mag[i][1:row_length-1])

	new_grad = np.array(new_grad)
	new_mag = np.array(new_mag)

	#print("\nGRADIENT")
	#print(new_grad)
	#print("\nMAGNITUDE")
	#print(new_mag)

	gradient_magnitude_stack = np.ravel(np.dstack((new_grad,new_mag)))

	return gradient_magnitude_stack


def create_grad_arrays_from_file(file_name):
	labels = []
	inputs = []
	with open(file_name, 'r') as f:
		reader = csv.reader(f)
		for count, row in enumerate(reader):
			array = np.array(row[1:])
			array = array.astype(np.float)
			labels.append(row[0])
			array = np.reshape(array,(5,4))
			grad_array = create_grad_array(array)
			#print("\nGRADIENT & MAGNITUDE STACK #" + str(count + 1))
			#print(grad_array)
			inputs.append(grad_array)
	
	return inputs, labels

if __name__ == '__main__':
	array = create_grad_array(image_array)
	print("\nGRADIENT & MAGNITUDE STACK")
	print(array)