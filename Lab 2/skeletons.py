import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import io
from skimage.morphology import skeletonize, thin
from skimage.color import rgb2gray
from skimage.transform import probabilistic_hough_line

data = "./input/bone_seg/"


def change_image_to_bw(path):
    img = Image.open(path)
    width = img.size[0]
    height = img.size[1]
    for i in range(0, width):
        for j in range(0, height):
            data = img.getpixel((i, j))
            if data[0] == 255 and data[1] == 0 and data[2] == 0:
                img.putpixel((i, j), (255, 255, 255))
            if data[0] == 0 and data[1] == 255 and data[2] == 0:
                img.putpixel((i, j), (255, 255, 255))
            if data[0] == 0 and data[1] == 0 and data[2] == 255:
                img.putpixel((i, j), (255, 255, 255))
            if data[0] == 32 and data[1] == 32 and data[2] == 32:
                img.putpixel((i, j), (0, 0, 0))
    return img


if __name__ == "__main__":
    os.mkdir("./results")
    os.mkdir("./results/lines")
    os.mkdir("./results/skeletons")
    os.mkdir("./results/bw_images")
    with open("./results/hallux_angles.txt", "w") as f:
        f.write("\n")

    for image in os.listdir(data):
        print(image)
        img = change_image_to_bw(f"{data}{image}")
        img.save(f"./results/bw_images/bw_{image}")
        img = rgb2gray(io.imread(f"./results/bw_images/bw_{image}"))

        # Metoda klasyczna
        skeletonized_img = skeletonize(img)
        plt.imshow(skeletonized_img)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"./results/skeletons/{image}")
        plt.clf()

        # Metoda lee
        skeletonized_lee_img = skeletonize(img, method='lee')
        plt.imshow(skeletonized_img)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"./results/skeletons/lee_{image}")
        plt.clf()

        # Metoda thin
        skeletonized_thin_img = thin(img)
        plt.imshow(skeletonized_thin_img)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"./results/skeletons/thin_{image}")
        plt.clf()

        # Linie aproksymujące szkielety tworzone metodą klasyczną
        angle = probabilistic_hough_line(skeletonized_img, threshold=15, line_length=110, line_gap=50)
        plt.imshow(skeletonized_img)
        plt.xticks([])
        plt.yticks([])
        for line in angle:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        plt.savefig(f"./results/lines/{image}")
        plt.clf()
        with open("./results/hallux_angles.txt", "a") as f:
            f.write(f"{image} - {angle}\n")

        # Linie aproksymujące szkielety tworzone metodą lee
        angle_lee = probabilistic_hough_line(skeletonized_lee_img, threshold=15, line_length=110, line_gap=50)
        plt.imshow(skeletonized_lee_img)
        plt.xticks([])
        plt.yticks([])
        for line in angle_lee:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        plt.savefig(f"./results/lines/lee_{image}")
        plt.clf()
        with open("./results/hallux_angles.txt", "a") as f:
            f.write(f"lee_{image} - {angle}\n")

        # Linie aproksymujące szkielety tworzone metodą thin
        angle_thinned = probabilistic_hough_line(skeletonized_thin_img, threshold=15, line_length=110, line_gap=50)
        plt.imshow(skeletonized_thin_img)
        plt.xticks([])
        plt.yticks([])
        for line in angle_thinned:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        plt.savefig(f"./results/lines/thin_{image}")
        plt.clf()
        with open("./results/hallux_angles.txt", "a") as f:
            f.write(f"thin_{image} - {angle}\n")
