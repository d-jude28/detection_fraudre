import streamlit as st
import pandas as pd
import joblib
import base64
from sklearn.preprocessing import OneHotEncoder
from openpyxl.workbook import Workbook
import matplotlib.pyplot as plt
import io

# Charger le modèle pré-entraîné
model = joblib.load('XGBoost.pkl')

# Variables utilisées dans l'encodage d'origine
original_columns = [
    'months_as_customer', 'age', 'policy_deductable', 'umbrella_limit',
    'incident_hour_of_the_day', 'number_of_vehicles_involved',
    'bodily_injuries', 'witnesses', 'total_claim_amount',
    'insured_sex_MALE', 'insured_education_level_College',
    'insured_education_level_High School', 'insured_education_level_JD',
    'insured_education_level_MD', 'insured_education_level_Masters',
    'insured_education_level_PhD', 'insured_occupation_armed-forces',
    'insured_occupation_craft-repair', 'insured_occupation_exec-managerial',
    'insured_occupation_farming-fishing', 'insured_occupation_handlers-cleaners',
    'insured_occupation_machine-op-inspct', 'insured_occupation_other-service',
    'insured_occupation_priv-house-serv', 'insured_occupation_prof-specialty',
    'insured_occupation_protective-serv', 'insured_occupation_sales',
    'insured_occupation_tech-support', 'insured_occupation_transport-moving',
    'incident_type_Parked Car', 'incident_type_Single Vehicle Collision',
    'incident_type_Vehicle Theft', 'collision_type_Rear Collision',
    'collision_type_Side Collision', 'incident_severity_Minor Damage',
    'incident_severity_Total Loss', 'incident_severity_Trivial Damage',
    'authorities_contacted_Fire', 'authorities_contacted_Other',
    'authorities_contacted_Police', 'authorities_contacted_nan',
    'property_damage_YES', 'police_report_available_YES'
]

# Fonction pour charger les données
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Format de fichier non supporté!")
                return None
            return data
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
            return None
    return None

def preprocess_data(df):
    # Colonnes à encoder
    columns_to_encode = ['insured_sex', 'insured_education_level', 'insured_occupation',
                         'incident_type', 'collision_type', 'incident_severity',
                         'authorities_contacted', 'property_damage', 'police_report_available']
    
    # Encoder uniquement les colonnes nécessaires
    enc = OneHotEncoder(handle_unknown='ignore', drop='first')
    cat_enc_data = pd.DataFrame(enc.fit_transform(df[columns_to_encode]).toarray(), columns=enc.get_feature_names_out(columns_to_encode))
    
    # Autres colonnes numériques
    num_data = df[['months_as_customer', 'age', 'policy_deductable', 'umbrella_limit',
                   'incident_hour_of_the_day', 'number_of_vehicles_involved',
                   'bodily_injuries', 'witnesses', 'total_claim_amount']]
    
    # Concaténer les données encodées avec les données numériques
    final_data = pd.concat([num_data.reset_index(drop=True), cat_enc_data.reset_index(drop=True)], axis=1)
    
    # Assurer que les colonnes sont les mêmes que celles du modèle
    final_data = final_data.reindex(columns=original_columns, fill_value=0)
    
    return final_data

def perform_prediction(data):
    # Vérifier que l'ordre des colonnes correspond à celui du modèle
    if list(data.columns) != original_columns:
        st.error("Les colonnes ne correspondent pas à celles du modèle entraîné.")
        return None
    predictions = model.predict(data)
    return predictions

# Fonction pour générer un lien de téléchargement
def get_binary_file_downloader_html(df, file_type='csv'):
    if file_type == 'csv':
        towrite = io.BytesIO()
        df.to_csv(towrite, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="resultats_fraude.csv">Télécharger les résultats en CSV</a>'
    else:
        towrite = io.BytesIO()
        df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="resultats_fraude.xlsx">Télécharger les résultats en Excel</a>'
    return href

# Interface Streamlit
def main():
    st.title("Détection de Fraude Sur les réclamations")
    # Uploader de fichier
    uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV ou Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Aperçu des données chargées:")
            st.write(data.head())

            # Préparer les données
            prepared_data = preprocess_data(data)
            if prepared_data is not None:
                st.write("Aperçu des données après préparation:")
                st.write(prepared_data.head())

                # Bouton pour effectuer la prédiction
                if st.button("Prédire les fraudes"):
                    predictions = perform_prediction(prepared_data)
                    if predictions is not None:
                        # Ajouter les prédictions aux données préparées
                        prepared_data['Prédiction'] = predictions
                        prepared_data['Prédiction'] = prepared_data['Prédiction'].map({1: 'Oui', 0: 'Non'})
                        
                        st.write("Résultats des prédictions:")
                        st.write(prepared_data.head())

                        # Visualiser le pourcentage de fraude avec un camembert
                        st.write("Pourcentage de fraude:")
                        fraud_percentage = prepared_data['Prédiction'].value_counts(normalize=True) * 100
                        fig, ax = plt.subplots()
                        ax.pie(fraud_percentage, labels=fraud_percentage.index, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')  # Assure que le camembert est bien circulaire
                        st.pyplot(fig)
                        
                        # Créer un lien de téléchargement pour le fichier de résultats
                        st.sidebar.markdown("### Télécharger le fichier:")
                        st.sidebar.markdown(get_binary_file_downloader_html(prepared_data, file_type='xlsx'), unsafe_allow_html=True)
                        st.sidebar.markdown(get_binary_file_downloader_html(prepared_data, file_type='csv'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
