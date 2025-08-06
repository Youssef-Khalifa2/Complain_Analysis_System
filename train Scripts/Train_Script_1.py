# train_model.py
import subprocess
import sys

# Ensure all libraries are installed
def install_packages():
    packages = [
        'pandas==1.5.3',
        'numpy==1.24.3',
        'tensorflow==2.13.0',
        'scikit-learn==1.3.0',
        'imbalanced-learn==0.11.0',
        'joblib==1.3.2',
        'langdetect==1.0.9',
        'transformers==4.34.0',
        'snowflake-connector-python==3.0.2'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages
install_packages()
# train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers, callbacks
import joblib
import re
from langdetect import detect, DetectorFactory
from transformers import BertTokenizer, TFBertModel

# Set seed for reproducibility
DetectorFactory.seed = 0

# Function definitions for data cleaning
def detect_language(text):
    try:
        return detect(text)
    except:
        return None

def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='tf', max_length=max_length)

def main():
    # Load data from Snowflake or local CSV (adapt as needed)
    conn = snowflake.connector.connect(
        user='ykhalifa',
        password='Youssef-2411',
        account='fq42011.west-europe.azure',
        warehouse='PROD_WH',
        database='PROPERTIES_DWH_EG',
        schema='VW_XXEMR_MONTH_COMPLAINTS_V'
    )
    cur = conn.cursor()
    sql = "SELECT * FROM PRODUCTION.PROPERTIES_DWH_EG.VW_XXEMR_MONTH_COMPLAINTS_V"
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    cur.close()
    conn.close()

    # Data filtering and cleaning
    df = df[(df['CALL_TYPE'] == "Complaint") & (~df['PROBLEM_CODE'].str.contains("Mobile", regex=False, na=False)) & df['PROBLEM_CODE'].notna()]
    df = df[['PROBLEM_SUMMARY', 'PROBLEM_CODE']]
    df['language'] = df['PROBLEM_SUMMARY'].apply(detect_language)
    df = df[df['language'] == 'en']
    df['PROBLEM_SUMMARY'] = df['PROBLEM_SUMMARY'].apply(clean_text)
    df = df.drop('language', axis=1)

    # BERT Embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    batch_size = 1024
    num_batches = (len(df) + batch_size - 1) // batch_size
    bert_embeddings = []

    for i in range(num_batches):
        batch_data = df['PROBLEM_SUMMARY'].iloc[i * batch_size:(i + 1) * batch_size]
        encoded_inputs = encode_texts(batch_data.tolist(), tokenizer)
        outputs = bert_model(encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'])
        bert_embeddings.append(outputs.pooler_output.numpy())

    bert_embeddings = np.concatenate(bert_embeddings)
    embeddings_df = pd.DataFrame(bert_embeddings)
    Y_target = df['PROBLEM_CODE']

    # Label encoding
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y_target)

    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_resampled, Y_resampled = smote.fit_resample(embeddings_df, Y_encoded)

    # Data splitting
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    # Model building
    model = models.Sequential([
        layers.InputLayer(input_shape=(X_train.shape[1],)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Model training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    # Save model
    model.save('dlearn_model.h5')
    joblib.dump(label_encoder, 'label_encoder.joblib')

if __name__ == "__main__":
    main()
