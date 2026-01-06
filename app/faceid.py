#import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#other dependencies
import cv2
import os
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

#import PyTorch model & layers
from layers import L1Dist
from model import EmbeddingNet, SiameseNetwork   # <-- your model files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CamApp(App):

    def build(self):
        # UI components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #pytorch model
        embedding = EmbeddingNet()
        self.model = SiameseNetwork(embedding)

        checkpoint = torch.load("siamese_model.pth", map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # Webcam
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        #preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()  # scales to [0,1] automatically
        ])

        return layout

    #webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr'
        )
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    #load and preprocess image
    def load_and_preprocess(self, path):
        img = PILImage.open(path).convert("RGB")
        img = self.preprocess(img)
        return img

    #verification logic
    def verify(self, *args):
        detection_threshold = 0.99
        verification_threshold = 0.8

        # Capture input image
        SAVE_PATH = os.path.join(
            'application_data', 'input_image', 'input_image.jpg'
        )
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []

        with torch.no_grad():  
            for image in os.listdir(
                os.path.join('application_data', 'verification_images')
            ):
                input_img = self.load_and_preprocess(
                    os.path.join('application_data', 'input_image', 'input_image.jpg')
                )
                validation_img = self.load_and_preprocess(
                    os.path.join('application_data', 'verification_images', image)
                )

                #add batch dimension
                input_img = input_img.unsqueeze(0).to(device)
                validation_img = validation_img.unsqueeze(0).to(device)

                #forward pass
                result = self.model(input_img, validation_img)
                results.append(result.item())

        #detection & verification logic
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(results)
        verified = verification > verification_threshold

        self.verification_label.text = "Verified" if verified else "Unverified"

        Logger.info(f"Results: {results}")
        Logger.info(f"Detection count: {detection}")
        Logger.info(f"Verification score: {verification}")
        Logger.info(f"Verified: {verified}")

        return results, verified


if __name__ == "__main__":
    CamApp().run()
