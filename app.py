from flask import Flask, request, render_template, send_from_directory, redirect
import os
from PIL import Image
from numpy.random import randint
import io
import numpy as np
from os import listdir
from PIL import Image
from torch_func import to_rgb, GeneratorResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


img_height = 256
# size of image width
img_width = 256
# number of image channels
channels = 3
# number of residual blocks in generator
n_residual_blocks = 9

batch_size = 4
n_workers = 8
debug_mode = False

input_shape = (channels, img_height, img_width)

# If gpu is available uncomment
# cuda = torch.cuda.is_available()
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

Tensor = torch.Tensor

data_transforms = transforms.Compose(
    [
        transforms.Resize(int(img_height * 1), Image.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def image_loader(loader, path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = to_rgb(image)
    image = loader(image).float()
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image


G_AB = GeneratorResNet(input_shape, n_residual_blocks)
# if cuda:
#     G_AB = G_AB.cuda()

G_AB.load_state_dict(torch.load("static/models/saved_models/G_AB.pth"))
G_AB.eval()

# G_BA = GeneratorResNet(input_shape, n_residual_blocks)
# # if cuda:
# #     G_AB = G_AB.cuda()

# G_BA.load_state_dict(torch.load("static/models/saved_models/G_BA.pth"))
# G_BA.eval()



# default access page
@app.route("/")
def main():
    return render_template("index.html")


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, "static/images/")

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return (
            render_template("error.html", message="The selected file is not supported"),
            400,
        )

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("processing.html", image_name=filename)


# flip filename 'vertical' or 'horizontal'
@app.route("/flip", methods=["POST"])
def flip():

    # retrieve parameters from html form
    if "horizontal" in request.form["mode"]:
        mode = "horizontal"
    elif "vertical" in request.form["mode"]:
        mode = "vertical"
    elif "reconstruction" in request.form["mode"]:
        mode = "reconstruction"
    elif "both" in request.form["mode"]:
        mode = "both"
    else:
        return render_template("error.html", message="Mode not supported"), 400
    filename = request.form["image"]

    # open and process image
    target = os.path.join(APP_ROOT, "static/images")
    destination = "/".join([target, filename])

    img = Image.open(destination)
    # data = load_image(destination)
    if mode == "vertical":
        # rename all A to B other than G_BA
        G_BA = GeneratorResNet(input_shape, n_residual_blocks)
        G_BA.load_state_dict(torch.load("static/models/saved_models/G_BA.pth"))
        G_BA.eval()
        real_B = image_loader(data_transforms, destination).type(Tensor)
        fake_A = G_BA(real_B)
        fake_A = make_grid(fake_A, nrow=1, normalize=True)
        real_B = make_grid(real_B, nrow=1, normalize=True)
        image_grid = torch.cat((real_B, fake_A), -1)
        save_image(image_grid, f"static/images/plot2.png", normalize=False)
        img = Image.open("static/images/plot2.png")

        # cust = {"InstanceNormalization": InstanceNormalization}
        # model_AtoB = load_model("static/models/g_model_AtoB_018060.h5", cust)
        # A_real = select_sample(data, 1)
        # B_generated = model_AtoB.predict(A_real)
        # b_gen = numpy.squeeze(B_generated)
        # img = tf.keras.preprocessing.image.array_to_img(
        #     b_gen, data_format=None, scale=True, dtype=None
        # )
    elif mode == "horizontal":
        real_A = image_loader(data_transforms, destination).type(Tensor)
        fake_B = G_AB(real_A)
        fake_B = make_grid(fake_B, nrow=1, normalize=True)

        real_A = make_grid(real_A, nrow=1, normalize=True)
        image_grid = torch.cat((real_A, fake_B), -1)
        save_image(image_grid, f"static/images/plot2.png", normalize=False)
        img = Image.open("static/images/plot2.png")

        # cust = {"InstanceNormalization": InstanceNormalization}
        # model_AtoB = load_model("static/models/g_model_AtoB_018060.h5", cust)
        # A_real = select_sample(data, 1)
        # B_generated = model_AtoB.predict(A_real)
        # show_plot2(A_real, B_generated)
        # img = Image.open("static/images/plot2.png")

    elif mode == "reconstruction":
        G_BA = GeneratorResNet(input_shape, n_residual_blocks)
        G_BA.load_state_dict(torch.load("static/models/saved_models/G_BA.pth"))
        G_BA.eval()
        real_A = image_loader(data_transforms, destination).type(Tensor)
        fake_B = G_AB(real_A)
        reel_A = G_BA(fake_B)
        real_A = make_grid(reel_A, nrow=1, normalize=True)
        fake_B = make_grid(fake_B, nrow=1, normalize=True)
        reel_A = make_grid(reel_A, nrow=1, normalize=True)
        image_grid = torch.cat((real_A, fake_B, reel_A), -1)
        save_image(image_grid, f"static/images/plot3.png", normalize=False)
        img = Image.open("static/images/plot3.png")

    else:
        return (
            render_template("error.html", message="Mode not supported :("),
            400,
        )
        filename = request.form["image"]
        # cust = {"InstanceNormalization": InstanceNormalization}
        # model_AtoB = load_model("static/models/g_model_AtoB_018060.h5", cust)
        # model_BtoA = load_model("static/models/g_model_BtoA_018060.h5", cust)
        # A_real = select_sample(data, 1)
        # B_generated = model_AtoB.predict(A_real)
        # A_generated = model_BtoA.predict(B_generated)
        # show_plot(A_real, B_generated, A_generated)
        # img = Image.open("static/images/plot3.png")

    # save and return image
    destination = "/".join([target, "temp.png"])
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    return send_image("temp.png")


# retrieve file from 'static/images' directory
@app.route("/static/images/<filename>")
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run()
