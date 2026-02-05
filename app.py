from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained model pipeline (includes preprocessing)
try:
    model = joblib.load('house_price_model.joblib')
    logger.info("✓ Model loaded successfully")
except FileNotFoundError:
    logger.error("ERROR: house_price_model.joblib not found. Please run Model_Training_and_Comparison.ipynb first.")
    model = None

# Define required features for prediction
REQUIRED_FEATURES = ['Location', 'Size', 'Bedrooms', 'Bathrooms', 'Condition', 'Type', 'Year Built', 'Date Sold']
NUMERIC_FEATURES = ['Size', 'Bedrooms', 'Bathrooms', 'Year Built']
CATEGORICAL_FEATURES = ['Location', 'Condition', 'Type']


def validate_input(data):
    """
    Validate the input data for required fields and data types.
    
    Args:
        data (dict): Input JSON data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check for required fields
    missing_fields = [field for field in REQUIRED_FEATURES if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate numeric fields
    for field in NUMERIC_FEATURES:
        try:
            value = float(data[field])
            if value < 0:
                return False, f"Field '{field}' cannot be negative. Provided: {value}"
        except (ValueError, TypeError):
            return False, f"Field '{field}' must be a number. Provided: {data[field]}"
    
    # Validate Bedrooms and Bathrooms (should be integers)
    for field in ['Bedrooms', 'Bathrooms']:
        try:
            value = int(float(data[field]))
            if value <= 0:
                return False, f"Field '{field}' must be greater than 0. Provided: {value}"
        except (ValueError, TypeError):
            return False, f"Field '{field}' must be a positive integer. Provided: {data[field]}"
    
    # Validate Size (should be positive)
    if float(data['Size']) <= 0:
        return False, "Size must be greater than 0 square feet"
    
    # Validate Year Built (should be reasonable)
    year_built = int(float(data['Year Built']))
    current_year = datetime.now().year
    if year_built > current_year:
        return False, f"Year Built cannot be in the future. Provided: {year_built}"
    if year_built < 1800:
        return False, f"Year Built seems too old. Provided: {year_built}"
    
    # Validate Date Sold (should be valid date and not in the future)
    try:
        date_sold = pd.to_datetime(data['Date Sold'])
        if date_sold > pd.Timestamp.now():
            return False, "Date Sold cannot be in the future"
    except (ValueError, TypeError):
        return False, f"Date Sold must be a valid date (YYYY-MM-DD format). Provided: {data['Date Sold']}"
    
    # Validate categorical fields (check if they're strings)
    for field in CATEGORICAL_FEATURES:
        if not isinstance(data[field], str) or len(data[field].strip()) == 0:
            return False, f"Field '{field}' must be a non-empty string. Provided: {data[field]}"
    
    return True, None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'ok', 'message': 'API is running'}), 200


@app.route('/info', methods=['GET'])
def info():
    """API information and usage guide"""
    return jsonify({
        'api_name': 'House Price Prediction API',
        'version': '1.0.0',
        'description': 'Predicts house prices based on real estate property features',
        'endpoints': {
            'POST /predict': 'Predict house price for a property',
            'GET /health': 'Health check',
            'GET /info': 'API information'
        },
        'required_fields': REQUIRED_FEATURES,
        'field_descriptions': {
            'Location': 'Geographical location of the property (string)',
            'Size': 'Property size in square feet (number)',
            'Bedrooms': 'Number of bedrooms (integer)',
            'Bathrooms': 'Number of bathrooms (integer)',
            'Condition': 'Condition of the property (string)',
            'Type': 'Property type/home type (string)',
            'Year Built': 'Year the property was built (integer)',
            'Date Sold': 'Date of sale in YYYY-MM-DD format (string)'
        },
        'example_request': {
            'Location': 'Downtown',
            'Size': 2500,
            'Bedrooms': 3,
            'Bathrooms': 2,
            'Condition': 'Good',
            'Type': 'House',
            'Year Built': 2010,
            'Date Sold': '2024-01-15'
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict house price based on property features.
    
    Expected JSON input with required fields:
    - Location, Size, Bedrooms, Bathrooms, Condition, Type, Year Built, Date Sold
    
    Returns:
    - JSON with predicted price or error message
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not available. Please check the server logs.'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided. Please send JSON data with required fields.'
            }), 400
        
        # Validate input
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_message
            }), 400
        
        # Create DataFrame with input data
        query_df = pd.DataFrame([data])
        
        # Convert Date Sold to datetime
        query_df['Date Sold'] = pd.to_datetime(query_df['Date Sold'])
        
        # Calculate Property Age
        query_df['Property_Age'] = query_df['Date Sold'].dt.year - query_df['Year Built']
        
        # Ensure proper data types for numeric features
        for field in NUMERIC_FEATURES:
            query_df[field] = pd.to_numeric(query_df[field], errors='coerce')
        
        # Check for any NaN values after conversion
        if query_df.isnull().any().any():
            return jsonify({
                'status': 'error',
                'message': 'Error processing input data. Please check all fields are valid.'
            }), 400
        
        # Make prediction
        prediction = model.predict(query_df)
        predicted_price = float(prediction[0])
        
        # Log the prediction
        logger.info(f"Prediction made: ${predicted_price:,.2f} for property in {data['Location']}")
        
        return jsonify({
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'message': f'Predicted price for the property is ${predicted_price:,.2f}',
            'input_features': data
        }), 200
    
    except pd.errors.ParserError as e:
        logger.error(f"Date parsing error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Invalid date format. Please use YYYY-MM-DD format. Error: {str(e)}'
        }), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'An error occurred during prediction: {str(e)}'
        }), 500


@app.errorhandler(400)
def bad_request(error):
    """Handle bad requests"""
    return jsonify({
        'status': 'error',
        'message': 'Bad request. Please check your input.'
    }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found. Use GET /info for available endpoints.'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error. Please try again later.'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION API")
    print("="*70)
    print("✓ API is starting...")
    print("✓ Available endpoints:")
    print("  - GET  http://localhost:5000/health     (Health check)")
    print("  - GET  http://localhost:5000/info       (API information & usage)")
    print("  - POST http://localhost:5000/predict    (Make predictions)")
    print("="*70 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
