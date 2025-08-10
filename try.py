import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from datetime import datetime, timedelta
import tensorflow as tf
import json
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DropoutPredictionModel:
    """Mô hình dự đoán nguy cơ bỏ học"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'days_since_last_activity',
            'average_session_duration',
            'completion_rate',
            'average_score',
            'study_streak',
            'total_time_spent',
            'lessons_per_week',
            'assessment_attempts',
            'forum_participation',
            'resource_downloads'
        ]
        
    def prepare_features(self, student_data: Dict) -> np.ndarray:
        """Chuẩn bị features từ dữ liệu sinh viên"""
        features = []
        
        # Days since last activity
        last_activity = pd.to_datetime(student_data.get('last_activity', datetime.now()))
        days_inactive = (datetime.now() - last_activity).days
        features.append(days_inactive)
        
        # Average session duration (minutes)
        features.append(student_data.get('avg_session_duration', 0) / 60)
        
        # Completion rate
        features.append(student_data.get('completion_rate', 0))
        
        # Average score
        features.append(student_data.get('average_score', 0))
        
        # Study streak
        features.append(student_data.get('study_streak', 0))
        
        # Total time spent (hours)
        features.append(student_data.get('total_time_spent', 0) / 3600)
        
        # Lessons per week
        features.append(student_data.get('lessons_per_week', 0))
        
        # Assessment attempts
        features.append(student_data.get('assessment_attempts', 0))
        
        # Forum participation
        features.append(student_data.get('forum_posts', 0))
        
        # Resource downloads
        features.append(student_data.get('resource_downloads', 0))
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: pd.DataFrame):
        """Train dropout prediction model"""
        # Prepare features and labels
        training_data = training_data.rename(columns={
            'forum_posts': 'forum_participation'
        })
        X = training_data[self.feature_names].values
        y = training_data['dropped_out'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Dropout prediction model accuracy: {accuracy:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance:")
        logger.info(feature_importance)
        
    def predict(self, student_data: Dict) -> Dict:
        """Dự đoán nguy cơ bỏ học"""
        if self.model is None:
            raise ValueError("Model chưa được train")
            
        features = self.prepare_features(student_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict probability
        dropout_prob = self.model.predict_proba(features_scaled)[0, 1]
        
        # Risk factors
        risk_factors = self._identify_risk_factors(student_data, features[0])
        
        return {
            'dropout_probability': float(dropout_prob),
            'risk_level': self._get_risk_level(dropout_prob),
            'risk_factors': risk_factors,
            'confidence': float(np.max(self.model.predict_proba(features_scaled)[0])),
            'recommendation': self._get_recommendation(dropout_prob, risk_factors)
        }
    
    def _identify_risk_factors(self, student_data: Dict, features: np.ndarray) -> List[Dict]:
        """Xác định các yếu tố rủi ro"""
        risk_factors = []
        
        if features[0] > 7:  # Days inactive
            risk_factors.append({
                'factor': 'Không hoạt động lâu',
                'severity': 'high',
                'value': f"{int(features[0])} ngày"
            })
            
        if features[2] < 30:  # Completion rate
            risk_factors.append({
                'factor': 'Tỷ lệ hoàn thành thấp',
                'severity': 'medium',
                'value': f"{features[2]:.1f}%"
            })
            
        if features[4] == 0:  # Study streak
            risk_factors.append({
                'factor': 'Không có streak học tập',
                'severity': 'medium',
                'value': "0 ngày"
            })
            
        return risk_factors
    
    def _get_risk_level(self, probability: float) -> str:
        """Xác định mức độ rủi ro"""
        if probability >= 0.8:
            return "very_high"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        elif probability >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def _get_recommendation(self, probability: float, risk_factors: List[Dict]) -> str:
        """Đưa ra khuyến nghị"""
        if probability >= 0.7:
            return "Cần can thiệp ngay! Liên hệ với học viên và đưa ra hỗ trợ cá nhân hóa."
        elif probability >= 0.5:
            return "Gửi email động viên và nhắc nhở về lộ trình học tập."
        elif probability >= 0.3:
            return "Theo dõi sát sao và gửi nội dung học tập hấp dẫn."
        else:
            return "Tiếp tục duy trì và khuyến khích học viên."


class PerformancePredictionModel:
    """Mô hình dự đoán hiệu suất học tập"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def build_neural_network(self, input_dim: int) -> keras.Model:
        """Xây dựng mạng neural network"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequence_features(self, learning_history: pd.DataFrame) -> np.ndarray:
        """Chuẩn bị features dạng sequence từ lịch sử học"""
        # Aggregate learning patterns over time
        features = []
        
        # Weekly patterns
        for week in range(4):  # Last 4 weeks
            week_data = learning_history[
                learning_history['week_number'] == week
            ]
            
            if not week_data.empty:
                features.extend([
                    week_data['study_hours'].sum(),
                    week_data['lessons_completed'].sum(),
                    week_data['average_score'].mean(),
                    week_data['forum_posts'].sum()
                ])
            else:
                features.extend([0, 0, 0, 0])
                
        # Trend features
        if len(learning_history) > 1:
            score_trend = np.polyfit(
                range(len(learning_history)),
                learning_history['average_score'].values,
                1
            )[0]
            features.append(score_trend)
        else:
            features.append(0)
            
        return np.array(features)
    
    def train(self, training_data: pd.DataFrame):
        """Train performance prediction model"""
        # Prepare features
        feature_list = []
        labels = []
        
        for student_id in training_data['student_id'].unique():
            student_data = training_data[training_data['student_id'] == student_id]
            
            if len(student_data) >= 4:  # Need at least 4 weeks of data
                features = self.prepare_sequence_features(student_data)
                final_score = student_data.iloc[-1]['final_score']
                
                feature_list.append(features)
                labels.append(final_score / 100.0)  # Normalize to 0-1
                
        X = np.array(feature_list)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        self.model = self.build_neural_network(X_train_scaled.shape[1])
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        test_loss, test_mae = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        logger.info(f"Performance prediction MAE: {test_mae:.3f}")
        
    def predict(self, learning_history: pd.DataFrame) -> Dict:
        """Dự đoán hiệu suất tương lai"""
        if self.model is None:
            raise ValueError("Model chưa được train")
            
        features = self.prepare_sequence_features(learning_history)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        predicted_score = self.model.predict(features_scaled, verbose=0)[0, 0] * 100
        
        # Confidence interval (simplified)
        confidence_margin = 5.0  # ±5%
        
        return {
            'predicted_score': float(predicted_score),
            'confidence_interval': {
                'lower': float(max(0, predicted_score - confidence_margin)),
                'upper': float(min(100, predicted_score + confidence_margin))
            },
            'trend': self._analyze_trend(learning_history),
            'strengths': self._identify_strengths(learning_history),
            'improvement_areas': self._identify_improvement_areas(learning_history)
        }
    
    def _analyze_trend(self, history: pd.DataFrame) -> str:
        """Phân tích xu hướng học tập"""
        if len(history) < 2:
            return "insufficient_data"
            
        scores = history['average_score'].values
        trend = np.polyfit(range(len(scores)), scores, 1)[0]
        
        if trend > 0.5:
            return "improving"
        elif trend < -0.5:
            return "declining"
        else:
            return "stable"
    
    def _identify_strengths(self, history: pd.DataFrame) -> List[str]:
        """Xác định điểm mạnh"""
        strengths = []
        
        avg_score = history['average_score'].mean()
        if avg_score >= 85:
            strengths.append("Điểm số xuất sắc")
            
        consistency = history['average_score'].std()
        if consistency < 5:
            strengths.append("Kết quả ổn định")
            
        completion_rate = history['lessons_completed'].sum() / history['total_lessons'].sum()
        if completion_rate >= 0.9:
            strengths.append("Hoàn thành đầy đủ bài học")
            
        return strengths
    
    def _identify_improvement_areas(self, history: pd.DataFrame) -> List[str]:
        """Xác định điểm cần cải thiện"""
        areas = []
        
        low_scores = history[history['average_score'] < 70]
        if len(low_scores) > len(history) * 0.3:
            areas.append("Cải thiện điểm số")
            
        study_time_var = history['study_hours'].std()
        if study_time_var > history['study_hours'].mean() * 0.5:
            areas.append("Duy trì thời gian học đều đặn")
            
        return areas


class RecommendationEngine:
    """Engine đề xuất nội dung học tập"""
    
    def __init__(self):
        self.content_embeddings = None
        self.user_embeddings = None
        
    def build_collaborative_filtering_model(self, num_users: int, num_items: int) -> keras.Model:
        """Xây dựng mô hình collaborative filtering"""
        # User embedding
        user_input = keras.layers.Input(shape=(1,))
        user_embedding = keras.layers.Embedding(num_users, 50)(user_input)
        user_vec = keras.layers.Flatten()(user_embedding)
        
        # Item embedding
        item_input = keras.layers.Input(shape=(1,))
        item_embedding = keras.layers.Embedding(num_items, 50)(item_input)
        item_vec = keras.layers.Flatten()(item_embedding)
        
        # Dot product
        dot_product = keras.layers.Dot(axes=1)([user_vec, item_vec])
        
        # Add bias terms
        user_bias = keras.layers.Embedding(num_users, 1)(user_input)
        user_bias = keras.layers.Flatten()(user_bias)
        
        item_bias = keras.layers.Embedding(num_items, 1)(item_input)
        item_bias = keras.layers.Flatten()(item_bias)
        
        # Combine
        output = keras.layers.Add()([dot_product, user_bias, item_bias])
        output = keras.layers.Activation('sigmoid')(output)
        
        model = keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def get_content_recommendations(self, student_id: str, 
                                  learning_history: pd.DataFrame,
                                  content_catalog: pd.DataFrame,
                                  n_recommendations: int = 5) -> List[Dict]:
        """Đề xuất nội dung học tập phù hợp"""
        
        # Analyze student preferences
        preferences = self._analyze_preferences(learning_history)
        
        # Filter suitable content
        suitable_content = self._filter_content(
            content_catalog,
            preferences,
            learning_history
        )
        
        # Rank content
        ranked_content = self._rank_content(
            suitable_content,
            preferences,
            student_id
        )
        
        # Generate recommendations
        recommendations = []
        for idx, content in ranked_content.head(n_recommendations).iterrows():
            recommendations.append({
                'content_id': content['id'],
                'title': content['title'],
                'type': content['type'],
                'difficulty': content['difficulty'],
                'estimated_time': content['duration'],
                'relevance_score': content['relevance_score'],
                'reason': self._get_recommendation_reason(content, preferences)
            })
            
        return recommendations
    
    def _analyze_preferences(self, history: pd.DataFrame) -> Dict:
        """Phân tích sở thích học tập"""
        preferences = {
            'preferred_difficulty': 'medium',
            'preferred_duration': 30,
            'preferred_types': [],
            'topics_of_interest': [],
            'learning_pace': 'normal'
        }
        
        if not history.empty:
            # Preferred difficulty
            avg_score = history['average_score'].mean()
            if avg_score >= 90:
                preferences['preferred_difficulty'] = 'hard'
            elif avg_score >= 75:
                preferences['preferred_difficulty'] = 'medium'
            else:
                preferences['preferred_difficulty'] = 'easy'
                
            # Preferred duration
            avg_duration = history['session_duration'].mean()
            preferences['preferred_duration'] = int(avg_duration)
            
            # Learning pace
            completion_speed = history['completion_time'].mean()
            if completion_speed < history['estimated_time'].mean() * 0.8:
                preferences['learning_pace'] = 'fast'
            elif completion_speed > history['estimated_time'].mean() * 1.2:
                preferences['learning_pace'] = 'slow'
                
        return preferences
    
    def _filter_content(self, catalog: pd.DataFrame, 
                       preferences: Dict,
                       history: pd.DataFrame) -> pd.DataFrame:
        """Lọc nội dung phù hợp"""
        # Filter by difficulty
        suitable = catalog[
            catalog['difficulty'] == preferences['preferred_difficulty']
        ].copy()
        
        # Filter out completed content
        completed_ids = history['content_id'].unique()
        suitable = suitable[~suitable['id'].isin(completed_ids)]
        
        # Filter by prerequisites
        # (Implementation depends on prerequisite tracking)
        
        return suitable
    
    def _rank_content(self, content: pd.DataFrame,
                     preferences: Dict,
                     student_id: str) -> pd.DataFrame:
        """Xếp hạng nội dung theo độ phù hợp"""
        content = content.copy()
        
        # Calculate relevance scores
        content['relevance_score'] = 0
        
        # Score by matching preferences
        content.loc[
            content['duration'] == preferences['preferred_duration'],
            'relevance_score'
        ] += 0.3
        
        # Score by popularity (simplified)
        content['relevance_score'] += content['popularity_score'] * 0.2
        
        # Score by freshness
        content['relevance_score'] += (
            1 - (datetime.now() - pd.to_datetime(content['created_at'])).dt.days / 365
        ) * 0.1
        
        # Sort by relevance
        return content.sort_values('relevance_score', ascending=False)
    
    def _get_recommendation_reason(self, content: pd.Series, 
                                  preferences: Dict) -> str:
        """Giải thích lý do đề xuất"""
        reasons = []
        
        if content['difficulty'] == preferences['preferred_difficulty']:
            reasons.append(f"Phù hợp với trình độ {preferences['preferred_difficulty']}")
            
        if content['popularity_score'] > 0.8:
            reasons.append("Được nhiều học viên đánh giá cao")
            
        if content.get('is_trending', False):
            reasons.append("Đang là xu hướng hot")
            
        return " | ".join(reasons) if reasons else "Phù hợp với lộ trình học tập"


class AIModelManager:
    """Quản lý tất cả các AI models"""
    
    def __init__(self):
        self.dropout_model = DropoutPredictionModel()
        self.performance_model = PerformancePredictionModel()
        self.recommendation_engine = RecommendationEngine()
        self.models_loaded = False
        
    def load_models(self, model_dir: str = "./models"):
        """Load pre-trained models"""
        try:
            # Load dropout model
            self.dropout_model.model = joblib.load(f"{model_dir}/dropout_model.pkl")
            self.dropout_model.scaler = joblib.load(f"{model_dir}/dropout_scaler.pkl")
            
            # Load performance model
            self.performance_model.model = keras.models.load_model(
                f"{model_dir}/performance_model.h5"
            )
            self.performance_model.scaler = joblib.load(f"{model_dir}/performance_scaler.pkl")
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
            
    def save_models(self, model_dir: str = "./models"):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save dropout model
        if self.dropout_model.model:
            joblib.dump(self.dropout_model.model, f"{model_dir}/dropout_model.pkl")
            joblib.dump(self.dropout_model.scaler, f"{model_dir}/dropout_scaler.pkl")
            
        # Save performance model
        if self.performance_model.model:
            self.performance_model.model.save(f"{model_dir}/performance_model.h5")
            joblib.dump(self.performance_model.scaler, f"{model_dir}/performance_scaler.pkl")
            
        logger.info("All models saved successfully")
        
    def predict_student_risk(self, student_data: Dict) -> Dict:
        """Dự đoán toàn diện về rủi ro của học viên"""
        # if not self.models_loaded:
        #     return {"error": "Models not loaded"}
            
        # Dropout risk
        dropout_prediction = self.dropout_model.predict(student_data)
        
        # Performance prediction
        if 'learning_history' in student_data:
            performance_prediction = self.performance_model.predict(
                student_data['learning_history']
            )
        else:
            performance_prediction = None
            
        # Combined risk assessment
        combined_risk = self._calculate_combined_risk(
            dropout_prediction,
            performance_prediction
        )
        
        return {
            'student_id': student_data.get('student_id'),
            'assessment_date': datetime.now().isoformat(),
            'dropout_risk': dropout_prediction,
            'performance_prediction': performance_prediction,
            'combined_risk_score': combined_risk['score'],
            'risk_category': combined_risk['category'],
            'intervention_priority': combined_risk['priority'],
            'recommended_actions': self._get_recommended_actions(combined_risk)
        }
    
    def _calculate_combined_risk(self, dropout_pred: Dict, 
                               perf_pred: Optional[Dict]) -> Dict:
        """Tính toán rủi ro tổng hợp"""
        risk_score = dropout_pred['dropout_probability'] * 0.6
        
        if perf_pred:
            # Low performance increases risk
            if perf_pred['predicted_score'] < 60:
                risk_score += 0.3
            elif perf_pred['predicted_score'] < 70:
                risk_score += 0.2
                
            # Declining trend increases risk
            if perf_pred['trend'] == 'declining':
                risk_score += 0.1
                
        # Determine category
        if risk_score >= 0.7:
            category = 'critical'
            priority = 1
        elif risk_score >= 0.5:
            category = 'high'
            priority = 2
        elif risk_score >= 0.3:
            category = 'medium'
            priority = 3
        else:
            category = 'low'
            priority = 4
            
        return {
            'score': float(risk_score),
            'category': category,
            'priority': priority
        }
    
    def _get_recommended_actions(self, risk_assessment: Dict) -> List[Dict]:
        """Đề xuất các hành động can thiệp"""
        actions = []
        
        if risk_assessment['category'] == 'critical':
            actions.extend([
                {
                    'action': 'immediate_contact',
                    'description': 'Liên hệ ngay với học viên qua điện thoại',
                    'deadline': '24 hours'
                },
                {
                    'action': 'personal_mentor',
                    'description': 'Chỉ định mentor 1-1 hỗ trợ',
                    'deadline': '48 hours'
                }
            ])
        elif risk_assessment['category'] == 'high':
            actions.extend([
                {
                    'action': 'personalized_email',
                    'description': 'Gửi email cá nhân hóa với lộ trình học tập',
                    'deadline': '2 days'
                },
                {
                    'action': 'study_group',
                    'description': 'Mời tham gia nhóm học tập phù hợp',
                    'deadline': '3 days'
                }
            ])
        elif risk_assessment['category'] == 'medium':
            actions.append({
                'action': 'motivation_content',
                'description': 'Gửi nội dung động viên và success stories',
                'deadline': '1 week'
            })
            
        return actions


# Real-time monitoring service
class RealTimeProgressMonitor:
    """Service theo dõi tiến độ real-time"""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.alert_thresholds = {
            'inactivity_days': 3,
            'low_score_threshold': 60,
            'completion_rate_drop': 20
        }
        
    def monitor_student_activity(self, student_id: str, 
                               activity_data: Dict) -> Optional[Dict]:
        """Theo dõi hoạt động và phát hiện anomaly"""
        alerts = []
        
        # Check inactivity
        if activity_data.get('days_since_last_activity', 0) > self.alert_thresholds['inactivity_days']:
            alerts.append({
                'type': 'inactivity',
                'severity': 'high',
                'message': f"Học viên không hoạt động {activity_data['days_since_last_activity']} ngày"
            })
            
        # Check performance drop
        if activity_data.get('latest_score', 100) < self.alert_thresholds['low_score_threshold']:
            alerts.append({
                'type': 'low_performance',
                'severity': 'medium',
                'message': f"Điểm số thấp: {activity_data['latest_score']}%"
            })
            
        # Check completion rate
        if activity_data.get('completion_rate_change', 0) < -self.alert_thresholds['completion_rate_drop']:
            alerts.append({
                'type': 'completion_drop',
                'severity': 'medium',
                'message': f"Tỷ lệ hoàn thành giảm {abs(activity_data['completion_rate_change'])}%"
            })
            
        if alerts:
            # Get AI risk assessment
            risk_assessment = self.model_manager.predict_student_risk({
                'student_id': student_id,
                **activity_data
            })
            
            return {
                'student_id': student_id,
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts,
                'risk_assessment': risk_assessment,
                'immediate_actions': self._get_immediate_actions(alerts, risk_assessment)
            }
            
        return None
    
    def _get_immediate_actions(self, alerts: List[Dict], 
                              risk_assessment: Dict) -> List[str]:
        """Đề xuất hành động cần làm ngay"""
        actions = []
        
        for alert in alerts:
            if alert['type'] == 'inactivity' and alert['severity'] == 'high':
                actions.append("Send re-engagement email với offer đặc biệt")
            elif alert['type'] == 'low_performance':
                actions.append("Gợi ý tài liệu bổ trợ và video giải thích")
                
        if risk_assessment.get('risk_category') == 'critical':
            actions.insert(0, "URGENT: Liên hệ trực tiếp với học viên")
            
        return actions


# Example usage
if __name__ == "__main__":
    # Initialize model manager
    model_manager = AIModelManager()
    
    # Simulate training data
    print("=== TRAINING AI MODELS ===")
    
    # Dropout prediction training data
    dropout_data = pd.DataFrame({
        'days_since_last_activity': np.random.randint(0, 30, 1000),
        'average_session_duration': np.random.normal(1800, 600, 1000),
        'completion_rate': np.random.uniform(0, 100, 1000),
        'average_score': np.random.normal(75, 15, 1000),
        'study_streak': np.random.randint(0, 30, 1000),
        'total_time_spent': np.random.uniform(0, 100000, 1000),
        'lessons_per_week': np.random.uniform(0, 10, 1000),
        'assessment_attempts': np.random.randint(0, 20, 1000),
        'forum_posts': np.random.randint(0, 50, 1000),
        'resource_downloads': np.random.randint(0, 30, 1000),
        'dropped_out': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })
    
    model_manager.dropout_model.train(dropout_data)
    
    print("\n=== MAKING PREDICTIONS ===")
    
    # Test student data
    test_student = {
        'student_id': 'user-student-01',
        'days_since_last_activity': 5,
        'avg_session_duration': 1200,
        'completion_rate': 45,
        'average_score': 68,
        'study_streak': 0,
        'total_time_spent': 15000,
        'lessons_per_week': 2,
        'assessment_attempts': 5,
        'forum_posts': 2,
        'resource_downloads': 8
    }
    
    # Predict risk
    risk_prediction = model_manager.predict_student_risk(test_student)
    print(json.dumps(risk_prediction, indent=2, ensure_ascii=False))
    
    # Real-time monitoring
    print("\n=== REAL-TIME MONITORING ===")
    monitor = RealTimeProgressMonitor(model_manager)
    
    alert_result = monitor.monitor_student_activity(
        'user-student-01',
        test_student
    )
    
    if alert_result:
        print(json.dumps(alert_result, indent=2, ensure_ascii=False))