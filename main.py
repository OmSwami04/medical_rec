from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Flask app
app = Flask(__name__)

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Symptoms dictionary
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 
    'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
    'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15,
    'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
    'weight_loss': 19, 'restlessness': 20, 'lethargy': 21,
    'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
    'sweating': 28, 'dehydration': 29, 'indigestion': 30,
    'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33,
    'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39,
    'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45,
    'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48,
    'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
    'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57,
    'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62,
    'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
    'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81,
    'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84,
    'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87,
    'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
    'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102,
    'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105,
    'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
    'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'blood_in_sputum': 117,
    'prominent_veins_on_calf': 118, 'palpitations': 119, 'painful_walking': 120,
    'pus_filled_pimples': 121, 'blackheads': 122, 'scurring': 123,
    'skin_peeling': 124, 'silver_like_dusting': 125, 'small_dents_in_nails': 126,
    'inflammatory_nails': 127, 'blister': 128, 'red_sore_around_nose': 129,
    'yellow_crust_ooze': 130
}


# Mapping of disease indices to disease names
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice',
    29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
    13: 'Dimorphic hemorrhoids (piles)', 18: 'Heart attack', 39: 'Varicose veins',
    26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthritis',
    5: 'Arthritis', 0: '(Vertigo) Paroxysmal Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

def extract_symptoms(user_input):
    """Extract symptoms from user input with improved matching."""
    user_input = user_input.lower()
    doc = nlp(user_input)

    # Extract words from input (force-match without lemmatization)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

    print(f"\n=== Debugging NLP Extraction ===")
    print(f"Raw Input: {user_input}")
    print(f"Tokens: {tokens}")
    print(f"===============================")

    # Check against symptom dictionary (manual matching)
    extracted_symptoms = []
    for word in tokens:
        for symptom in symptoms_dict.keys():
            if word in symptom or symptom in word:
                extracted_symptoms.append(symptom)

    print(f"Recognized Symptoms: {extracted_symptoms}")
    return extracted_symptoms





def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(132)  # Ensure input vector has 132 features

    # Get correct feature names from the trained model
    feature_names = list(svc.feature_names_in_)  # Ensure it matches the training data

    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            index = symptoms_dict[symptom]
            input_vector[index] = 1  # Mark symptom as present

    # Convert input vector into a DataFrame with correct shape
    input_df = pd.DataFrame([input_vector], columns=feature_names)

    # Make prediction
    predicted_index = int(svc.predict(input_df)[0])

    # Debugging: Print predicted values
    print(f"Predicted Index: {predicted_index}")
    print(f"Feature Names Count: {len(feature_names)}")
    print(f"Input Vector Shape: {input_df.shape}")

    # Convert index to disease name
    predicted_disease = diseases_list.get(predicted_index, "Unknown Disease")

    # Debugging: Print final disease name
    print(f"Predicted Disease: {predicted_disease}")

    return str(predicted_disease)  # Ensure it returns a string

def helper(dis):
    """Fetch disease details safely to avoid IndexError."""
    
    if dis not in description['Disease'].values:
        return "No description available", [], [], [], []
    
    desc_row = description[description['Disease'] == dis]
    desc = desc_row['Description'].values[0] if not desc_row.empty else "No description available"
    
    pre_row = precautions[precautions['Disease'] == dis]
    pre = pre_row[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten() if not pre_row.empty else []
    
    med_row = medications[medications['Disease'] == dis]
    med = med_row['Medication'].values if not med_row.empty else []
    
    diet_row = diets[diets['Disease'] == dis]
    die = diet_row['Diet'].values if not diet_row.empty else []
    
    workout_row = workout[workout['disease'] == dis]
    wrkout = workout_row['workout'].values if not workout_row.empty else []
    
    return desc, pre, med, die, wrkout

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms_text = request.form.get('symptoms')
        user_symptoms = extract_symptoms(symptoms_text)

        if not user_symptoms:
            return render_template('index.html', message="No recognizable symptoms found.")

        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        return render_template('index.html', 
                               predicted_disease=predicted_disease,
                               dis_des=dis_des,
                               my_precautions=precautions,
                               medications=medications,
                               my_diet=rec_diet,
                               workout=workout)

if __name__ == '__main__':
    app.run(debug=True)