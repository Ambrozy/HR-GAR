import os
import torchvision.io as io
import torchvision.transforms as T
from PIL import Image


def save_image(image, path):
    im = image.permute(1, 2, 0).numpy()
    im = Image.fromarray(im)
    im.save(path)


def prepare_flower_dataset(img_folder, output_folder, x_size=32, y_size=128):
    y_transforms = T.Compose([
        T.Resize(y_size),
        T.RandomHorizontalFlip(),
        T.RandomCrop(y_size),
    ])
    x_transforms = T.Compose([
        T.Resize(x_size),
    ])

    os.makedirs(output_folder + 'y/', exist_ok=True)
    os.makedirs(output_folder + 'x/', exist_ok=True)

    for name in os.listdir(img_folder):
        output_name = os.path.splitext(name)[0]
        if name.endswith('.jpg'):
            image = io.read_image(img_folder + name)
            y_image = y_transforms(image)
            x_image = x_transforms(y_image)

            save_image(y_image, output_folder + 'y/' + output_name + '.png')
            save_image(x_image, output_folder + 'x/' + output_name + '.png')


if __name__ == '__main__':
    IMG_FOLDER = '../Flowers/17flowers/jpg/'
    OUTPUT_FOLDER = '../Flowers/17flowers/'
    X_SIZE = 32
    Y_SIZE = 128

    prepare_flower_dataset(IMG_FOLDER, OUTPUT_FOLDER, X_SIZE, Y_SIZE)
