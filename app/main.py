import imp
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic

from PyQt5 import QtGui
from PIL.ImageQt import ImageQt

import random
import torch
import os

from PIL import Image
import torchvision.transforms.functional as TF

import numpy as np


class Main(QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("template.ui", self)

        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        
        self.select_model.currentIndexChanged.connect(self.set_model)
        self.set_model(0)

        self.random_btn.clicked.connect(lambda _: self.predict_random_image(self.c))
        self.upload_btn.clicked.connect(lambda _: self.predict_upload_image(self.c))
        self.save_btn.clicked.connect(lambda _: self.save())
    
    def set_model(self, index):
        c = self.select_model.itemText(index)
        
        if c == "Edges2Handbags":
            model_path = f"../weights/edges2handbags.jit.pt"
        elif c == "Facades":
            model_path = f"../weights/facades.jit.pt"
        elif c == "Maps":
            model_path = f"../weights/maps.jit.pt"
        else:
            return
        
        self.model = torch.jit.load(model_path)
        print(f"{c} model loaded!")

        self.c = c

        self.predict_random_image(c)
    
    def predict_random_image(self, c):
        img_name = random.choice(os.listdir("data_samples/" + c + "/"))
        img = Image.open("data_samples/" + c + "/" + img_name).convert("RGB").resize((128, 128))
        self.input_image.setPixmap(QPixmap(QImage(ImageQt(img.convert("RGBA")))))

        out_img = self.predict(img)
        self.output_image.setPixmap(QPixmap(QImage(ImageQt(out_img.convert("RGBA")))))
    
    def predict_upload_image(self, c):
        img_path = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        img = Image.open(img_path).convert("RGB").resize((128, 128))

        self.input_image.setPixmap(QPixmap(QImage(ImageQt(img.convert("RGBA")))))

        out_img = self.predict(img)
        self.output_image.setPixmap(QPixmap(QImage(ImageQt(out_img.convert("RGBA")))))
    
    def predict(self, img):
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)

        out = self.model(img[None])[0]

        out_img = TF.normalize(
            out, [-m/s for m, s in zip(self.mean, self.std)],
            [1/s for s in self.std]
        )

        self.out_img = TF.to_pil_image(torch.clamp(out_img, 0.0, 1.0))
        return self.out_img
    
    def save(self):
        name = QFileDialog.getSaveFileName(self, 'Save Image', "main.png")[0]
        self.out_img.save(name)

        print("Image saved!")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()

    ex.show()
    sys.exit(app.exec_())
