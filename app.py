from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline (includes preprocessing)
model = joblib.load('house_price_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400

        query_df = pd.DataFrame([data])

        # Ensure required columns exist
        if 'Year Built' not in query_df.columns:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: Year Built'
            }), 400

        # Handle Date Sold safely
        query_df['Date Sold'] = pd.to_datetime(
            query_df.get('Date Sold', pd.Timestamp.now())
        )

        query_df['Property_Age'] = (
            query_df['Date Sold'].dt.year - query_df['Year Built']
        )

        prediction = model.predict(query_df)

        return jsonify({
            'status': 'success',
            'predicted_price': round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
