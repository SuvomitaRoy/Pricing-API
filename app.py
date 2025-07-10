from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from dotenv import load_dotenv
import os
import json
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

load_dotenv()

app = Flask(__name__)

# --- Database connection details ---
DB_CONFIG = os.getenv("DB_CONFIG")
ENGINE_PG = os.getenv("ENGINE_PG")

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_client_encoding('utf8')
    return conn

# def insert_encoded_data_to_db(data, table_name = "ao_pricing_data"):
#     df = pd.DataFrame([data]) #problem: when converting to dataframe
#     # df_encoded = pd.get_dummies(df, drop_first=False)
#     # df_encoded[df_encoded.columns] = df_encoded[df_encoded.columns].astype(int)
#     try:
#         engine = create_engine(ENGINE_PG, connect_args={'client_encoding': 'utf8'})
#         df.to_sql(table_name, engine, if_exists='replace', index=False)
#         print("Data inserted successfully.")
#     except Exception as e:
#         print(f"Database insertion failed: {e}")

def load_data_from_db(table_name='ao_pricing_data'):
    try:
        engine = create_engine(
            ENGINE_PG,
            connect_args={'client_encoding': 'latin1'}
        )
        df = pd.read_sql_table(table_name, engine)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
    
def insert_encoded_data_to_db(data, table_name="ao_pricing_data"):
    def safe_encode(value):
        if isinstance(value, str):
            # Try multiple encoding strategies
            try:
                return value.encode('utf-8').decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return value.encode('latin-1').decode('utf-8', errors='replace')
                except:
                    return str(value).encode('utf-8', errors='replace').decode('utf-8')
        return value
    
    # Clean the data recursively
    def clean_data(obj):
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_data(item) for item in obj]
        else:
            return safe_encode(obj)
    
    cleaned_data = clean_data(data)
    df = pd.DataFrame([cleaned_data])
    
    try:
        engine = create_engine(ENGINE_PG, connect_args={'client_encoding': 'utf8'})
        df.to_sql('ao_pricing_data', engine, if_exists='replace', index=False)
        print("Data inserted successfully.")
    except Exception as e:
        print(f"Database insertion failed: {e}")
        # Optional: print the problematic data for debugging
        print(f"Problematic data: {data}")

def lasso_pricing():
    # Fetching and preparing the dataset
    df = load_data_from_db()
    y = df["Price Estimate (€)"]
    X = df.drop("Price Estimate (€)",axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Initialise and train Lasso model
    lasso_model = Lasso(alpha=1.0)
    lasso_model.fit(X_train, y_train)

    # Make predictions
    y_pred = lasso_model.predict(X_test)

    # Compute R² score
    r2 = r2_score(y_test, y_pred)
    print(f"Lasso Regression R² Score: {r2:.4f}")

    coef_df = pd.Series(lasso_model.coef_, index=X_train.columns)
    print("Finished calculating!")
    return coef_df

@app.route('/process-request', methods=['POST'])
def process_request():
    data = request.get_json()

    required_fields = ['ao_id', 'client_id', 'work_type', 'pricing_type', 'status', 'inputs']
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    try:
        inputs = data['inputs']
        insert_encoded_data_to_db(data['inputs'])
        coef_df = lasso_pricing()

        conn = get_db_connection()
        cur = conn.cursor()

        insert_sql = """
        INSERT INTO ao_pricing_model (
            ao_id, model_coeff, work_type, pricing_type, status, date_added, date_updated
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        if data['status'] == "ACTIVE":
            cur.execute(insert_sql, (
                data['ao_id'],
                json.dumps(coef_df.to_dict()),
                data['work_type'],
                data['pricing_type'],
                data['status'],
                datetime.utcnow(),
                None
            ))

        elif data['status'] == "INACTIVE":
            cur.execute(insert_sql, (
                data['ao_id'],
                json.dumps(coef_df.to_dict()),
                data['work_type'],
                data['pricing_type'],
                data['status'],
                datetime.utcnow(),
                datetime.utcnow()
            ))

        conn.commit()
        cur.close()
        conn.close()

        missing_inputs = [key for key in coef_df if key not in inputs]
        if missing_inputs:
            return jsonify({'error': f'Missing input values: {", ".join(missing_inputs)}'}), 400

        # Calculate weighted score
        try:
            total_score = sum(coef_df[key] * inputs[key] for key in coef_df)
        except Exception as e:
            return jsonify({'error': f'Error in input data: {str(e)}'}), 400

        return jsonify({'Price': total_score})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
