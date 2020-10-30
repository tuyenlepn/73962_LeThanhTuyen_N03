from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import glob, os

datagen = ImageDataGenerator(
        brightness_range=[0.1,1.0],
        rotation_range=40,
        zoom_range=[1.0,3.0],
        horizontal_flip=True
        )
os.chdir("IDCard")
for file in glob.glob("*.jpg"):
    img = load_img(file)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = np.expand_dims(x,0)  # this is a Numpy array with shape (1, 3, 150, 150)



    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='id', save_format='jpg'):
        i += 1
        if i > 10:
            break







