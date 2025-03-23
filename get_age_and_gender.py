# Global variables
IMG_P = "person_13_body.png"

# all package-imports
import cv2
import torch
from mivolo.predictor import Predictor

class Config:
    detector_weights = "models/yolov8x_person_face.pt"
    device = "cuda:0"
    checkpoint = "models/mivolo_imbd.pth.tar"
    with_persons = True
    disable_faces = True
    draw = False


def main():
    args = Config()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    predictor = Predictor(args, verbose=True)

    # load image and get final prediction
    img = cv2.imread(IMG_P)
    detected_objects, out_im, age, gender = predictor.recognize(img)

    print(f"We detected a person of age {age} and gender {gender}")


if __name__ == '__main__':
    main()