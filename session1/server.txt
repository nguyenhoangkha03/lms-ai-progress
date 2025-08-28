from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback
from datetime import datetime
import os
import sys

from exampel import (
    DatabaseManager, TestAnalyzer, LearningStrategyAI, 
    ContentRecommender, Learning_assessment, RandomForestLearninAttube,
    AITrackingDataCollector, AITRACKING
)

app = Flask(__name__)
CORS(app)  


db_manager = None
test_analyzer = None
learning_strategy_ai = None
content_recommender = None
learning_assessment = None  
random_forest_model = None
ai_tracking_collector = None

def initialize_models():
    """Khởi tạo tất cả models và database connection"""
    global db_manager, test_analyzer, learning_strategy_ai, content_recommender
    global learning_assessment, random_forest_model, ai_tracking_collector
    
    try:
        # Database connection
        db_manager = DatabaseManager()
        conn = db_manager.connect()
        
        if not conn:
            raise Exception("Không thể kết nối database")
            
        # Initialize models
        test_analyzer = TestAnalyzer()
        learning_strategy_ai = LearningStrategyAI()
        content_recommender = ContentRecommender()
        learning_assessment = Learning_assessment(db_manager)
        random_forest_model = RandomForestLearninAttube()
        ai_tracking_collector = AITrackingDataCollector(db_manager)
        ai_tracking = AITRACKING(db_manager)
        model_list = [ai_tracking, random_forest_model, learning_strategy_ai]
        
        try:
            if os.path.exists("./models/"):
                loaded_strategy = learning_strategy_ai.load_model("./models/learningstrategyai_model.joblib")
                loaded_attitude = random_forest_model.load_model("./models/attitude_model.joblib")       
                load_ai_track = ai_tracking.load_model("./models/aitrack_model.joblib")
               
             
                if loaded_strategy.is_trained:
                    learning_strategy_ai = loaded_strategy
                if loaded_attitude.is_trained:
                    random_forest_model = loaded_attitude
                if load_ai_track.is_trained:
                    ai_tracking = load_ai_track
                print("Pre-trained models loaded successfully")
        except Exception as e:
            print(f"Could not load pre-trained models: {e}")
          
            print("Training models...")
            learning_strategy_ai.train_model()
            learning_strategy_ai.save_model()
            random_forest_model.train()
            random_forest_model.save_model()
            ai_tracking.train_performance_model()
            ai_tracking.save_model()
            
            # Save trained models
            # ModelManager.save_all_models([learning_strategy_ai, random_forest_model, ai_tracking], "./models/")
            print("Models trained and saved")
            
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database_connected': db_manager is not None,
        'models_loaded': all([
            learning_strategy_ai is not None,
            random_forest_model is not None
        ])
    })

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Đề xuất lessons cho user"""
    try:
        data = request.get_json()
        questions = data.get('data')
        
        if not questions:
            return jsonify({'error': 'thiếu danh sách câu hỏi'}), 400
            
        # Get performance analysis
        # analysis = test_analyzer.analyze_user_performance(db_manager, user_id, assessment_id)
        analysis_begin = test_analyzer.analyze_user_begining(db_manager,questions)
        # print(analysis_begin)
        # Extract features for strategy prediction
        features = learning_strategy_ai.extract_features(analysis_begin)
        # print(features)
        # Predict strategy
        strategy, confidence, additional_info = learning_strategy_ai.predict_strategy(features)
        
        
        # Get lesson recommendations
        recommendations = content_recommender.recommend_lessons(db_manager, strategy, analysis_begin)
        # print("Con cá con:", analysis_begin)
        
        return jsonify({
            'success': True,
            'data': {
                'strategy': strategy,
                'strategy_confidence': confidence,
                'recommendations': recommendations,
                'total_recommendations': len(recommendations)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/recommend-lessons', methods=['POST'])
def recommend_lessons():
    """Đề xuất lessons cho user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        assessment_attemp_id = data.get('assessment_attemp_id')
        
        if not user_id or not assessment_attemp_id:
            return jsonify({'error': 'user_id và assessment_id là required'}), 400
            

        analysis = test_analyzer.analyze_user_performance(db_manager, user_id, assessment_attemp_id)
   
        features = learning_strategy_ai.extract_features(analysis)
        
        # Predict strategy
        strategy, confidence, additional_info = learning_strategy_ai.predict_strategy(features)
        
        
    
        recommendations = content_recommender.recommend_lessons(db_manager, strategy, analysis)
        # print("Con cá con:", analysis_begin)
        
        return jsonify({
            'success': True,
            'data': {
                'strategy': strategy,
                'strategy_confidence': confidence,
                'recommendations': recommendations,
                'total_recommendations': len(recommendations)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/predict-attitude', methods=['POST'])
def predict_learning_attitude():
    """Dự đoán learning attitude của user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id là required'}), 400
            
        
        analytics_data = learning_assessment.learning_analytics_data(user_id)
        
       
        attitude_result = random_forest_model.predict_attitude(analytics_data, return_proba=True)
        
        return jsonify({
            'success': True,
            'data': {
                'predicted_attitude': attitude_result['attitude'],
                'confidence': attitude_result['confidence'],
                'reason': attitude_result.get('reason', {}),
                'user_id': user_id,
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/comprehensive-analysis', methods=['POST'])
def comprehensive_analysis():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        assessment_id = data.get('assessment_id')
        lesson_id = data.get('lesson_id', 'lesson-html-tags')
        
        if not user_id or not assessment_id:
            return jsonify({'error': 'user_id và assessment_id là required'}), 400
            
        performance_analysis = test_analyzer.analyze_user_performance(db_manager, user_id, assessment_id)
        
      
        features = learning_strategy_ai.extract_features(performance_analysis)
        strategy, strategy_confidence, strategy_info = learning_strategy_ai.predict_strategy(features)
        
       
        recommendations = content_recommender.recommend_lessons(db_manager, strategy, performance_analysis)
        
        
        try:
            analytics_data = learning_assessment.learning_analytics_data(user_id, lesson_id)
            attitude_result = random_forest_model.predict(analytics_data, return_proba=True)
        except Exception as e:
            attitude_result = {
                'attitude': 'Unknown',
                'confidence': 0.0,
                'error': str(e)
            }
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'assessment_id': assessment_id,
                'lesson_id': lesson_id,
                'performance_analysis': performance_analysis,
                'strategy_prediction': {
                    'strategy': strategy,
                    'confidence': strategy_confidence,
                    'additional_info': strategy_info
                },
                'lesson_recommendations': recommendations,
                'attitude_prediction': attitude_result
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# @app.route('/api/models/info', methods=['GET'])
# def get_models_info():
#     """Lấy thông tin về các models"""
#     try:
#         model_info = ModelManager.create_model_info("./models/")
        
#         return jsonify({
#             'success': True,
#             'data': {
#                 'model_info': model_info,
#                 'runtime_status': {
#                     'learning_strategy_ai_trained': learning_strategy_ai.is_trained if learning_strategy_ai else False,
#                     'random_forest_attitude_trained': random_forest_model.is_trained if random_forest_model else False,
#                     'database_connected': db_manager is not None
#                 }
#             },
#             'timestamp': datetime.now().isoformat()
#         })
        
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e),
#             'traceback': traceback.format_exc()
#         }), 500

@app.route('/api/models/retrain', methods=['POST'])
def retrain_models():
    """Retrain tất cả models"""
    try:
        global learning_strategy_ai, random_forest_model
        
        print("Starting model retraining...")
        
     
        learning_strategy_ai.train_model()
        print("LearningStrategyAI retrained")
        learning_strategy_ai.save_model()
        
        # Retrain RandomForestLearninAttube
        random_forest_model.train()
        print("RandomForestLearninAttube retrained")
        random_forest_model.train()
        
        print("Retrained models saved")
        
        return jsonify({
            'success': True,
            'message': 'All models retrained and saved successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/tracking/collect', methods=['POST'])
def collect_tracking_data():
    """Thu thập dữ liệu AI tracking cho user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        course_id = data.get('course_id')
        
        if not user_id:
            return jsonify({'error': 'user_id là required'}), 400
            
        # Collect comprehensive tracking data
        tracking_data = ai_tracking_collector.collect_comprehensive_data(user_id, course_id)
        
        return jsonify({
            'success': True,
            'data': tracking_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health',
            '/api/analyze-performance',
            '/api/predict-strategy', 
            '/api/recommend-lessons',
            '/api/predict-attitude',
            '/api/comprehensive-analysis',
            '/api/models/info',
            '/api/models/retrain',
            '/api/tracking/collect'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500

if __name__ == '__main__':
    print("Starting AI Learning System Server...")
    
    # Initialize models
    if initialize_models():
        print("Models initialized successfully")
        print("Server starting on http://localhost:5000")
        print("\nAvailable endpoints:")
        print("  GET  /health                    - Health check")
        print("  POST /api/recommend     - Get lesson recommendations")
        print("  POST /api/recommend-lessons     - Get lesson recommendations")
        print("  POST /api/predict-attitude      - Predict learning attitude")
        print("  POST /api/comprehensive-analysis - Complete analysis")
        print("  GET  /api/models/info           - Get models information")
        print("  POST /api/models/retrain        - Retrain all models")
        print("  POST /api/tracking/collect      - Collect tracking data")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize models. Server not started.")
        sys.exit(1)