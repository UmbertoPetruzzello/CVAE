import matplotlib.pyplot as plt
import numpy as np


# helper function to un-normalize and display an image
def imshow(img, file_name = "reconstructed_image_conv2.png"):
    #img = img / 2 + 0.5  # unnormalize
    img = np.transpose(img, (1,2,0))
    img = img[:, :, 0]
    plt.imshow(img, cmap="gray")
    plt.savefig(file_name, bbox_inches = "tight", pad_inches = 0.0)
    print("done!")
