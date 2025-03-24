import sys

import specified_helper_functions as helper
import prediction_handling

# Configure logging
import logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def age_and_gender_classification(anon_type):
    # Set up 
    sist_basis = helper.get_anon_type_sist_basis(anon_type)
    all_unedited_cities = helper.collect_all_footage_dfs(anon_type)
    predictor = prediction_handling.set_up_predictor()

    age_data, gender_data = prediction_handling.predict_all_cities(all_unedited_cities, predictor)

    new_column_data_dict = {
        "age": age_data,
        "gender": gender_data
    }

    helper.update_sist_df(sist_basis, new_column_data_dict, anon_type)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        age_and_gender_classification(sys.argv[1])
    else:
        logging.warning(f"Age and Gender Classifier weren't able to be executed due to missing anon_type argument!")