import torch
import cv2
from mivolo.predictor import Predictor

# Global variables
class Config:
    detector_weights = "attribute_identifier/age_and_gender_classifier/models/yolov8x_person_face.pt"
    device = "cuda:0"
    checkpoint = "attribute_identifier/age_and_gender_classifier/models/mivolo_imbd.pth.tar"
    with_persons = True
    disable_faces = True
    draw = False

def set_up_predictor():
    args = Config()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return Predictor(args, verbose=True)

def _classify_age(age: float) -> str:
    """
    Groups ages into age-gaps of 5 years.
    """
    lower_bound = (int(age) // 5) * 5
    upper_bound = lower_bound + 4
    return f"{lower_bound}-{upper_bound}"

def predict_all_cities(all_cities_df, predictor):
    age_data_dict = {}
    gender_data_dict = {}
    for city in all_cities_df:
        for index, row in all_cities_df[city].iterrows():
            img = cv2.imread(row["body_crop_img_path"])
            detected_objects, out_im, age, gender = predictor.recognize(img)

            age_classified = _classify_age(age)
            age_data_dict[row["person_id"]] = age_classified
            gender_data_dict[row["person_id"]] = gender

    return age_data_dict, gender_data_dict