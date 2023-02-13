import numpy as np
import png
import pydicom

ds = pydicom.dcmread("/home/tkrsh/demo/images/0009.DCM")

shape = ds.pixel_array.shape

image_2d = ds.pixel_array.astype(float)

image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

image_2d_scaled = np.uint8(image_2d_scaled)

destination = "/home/tkrsh/demo/images/dicom.jpeg"
with open(destination, 'wb') as png_file:
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)
