import pymysql
import numpy as np
import pandas as pd
import calendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import json
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class DatabaseManager:
    """Quản lý kết nối và thao tác database"""

    def __init__(self):
        self.connection = None

    def connect(self):
        """Kết nối đến database"""
        try:
            self.connection = pymysql.connect(
                host='localhost',
                user='root',
                password='',
                database='lms_ai_database',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("✅ Kết nối thành công đến cơ sở dữ liệu!")
            return self.connection
        except pymysql.MySQLError as e:
            print(f"❌ Lỗi kết nối cơ sở dữ liệu: {e}")
            return None

    def select(self, nameTable, columns="*", option=None):
        """Truy vấn SELECT với điều kiện tùy chọn (WHERE)"""
        try:
            with self.connection.cursor() as cursor:
                if option:
                    sql = f"SELECT {columns} FROM {nameTable} {option}"
                else:
                    sql = f"SELECT {columns} FROM {nameTable}"

                # # In câu truy vấn SQL ra để debug
                # print("\n🔍 SQL đang chạy:")
                # print(sql)

                cursor.execute(sql)
                return cursor.fetchall()
        except Exception as e:
            print(f"❌ Lỗi SELECT: {e}")
            return []

    def close(self):
        """Đóng kết nối"""
        if self.connection:
            self.connection.close()
            print("🔐 Đã đóng kết nối database")

class TestAnalyzer:
    def analyze_user_performance(self, db: DatabaseManager, user_id: str, assessment_attempId : str):
        weak_categories = []
        avd_categories = []
        difficulty_stats = []
        question_ids = []
        seen_qids = set()
        lesson_recommendations = []  
        difficulty_performance = {}
        
        where_clause = f"""
        JOIN assessments a ON ast.assessmentId = a.id 
        LEFT JOIN lessons ls ON a.lessonId = ls.id
        LEFT JOIN courses cr ON cr.id = COALESCE(a.courseId, ls.courseId)
        LEFT JOIN categories cate ON cr.categoryId = cate.id 
        WHERE ast.studentId = '{user_id}' AND ast.id ='{assessment_attempId}' 
        """

        assessment_attempts = db.select(
            "assessment_attempts ast",
            """
            ast.*,a.id AS assessmentId, a.*,
            cate.name, cr.title AS course_title, ls.id AS lesson_id,
            ls.title AS lesson_title, ls.slug AS lesson_slug
            
            
            """,
            where_clause
        )
   
        df = pd.DataFrame(assessment_attempts)
        # if df.empty:
        #     return {'has_data': False}

        # df['course_title'] = df.get('course_title', 'Unknown Course').fillna('Unknown Course')
        # df['lesson_title'] = df.get('lesson_title', 'Unknown Lesson').fillna('Unknown Lesson')

        assessment_attempt = df[["id","assessmentId","score", "maxScore", "name", "course_title"]]
        questions = {}
        for _, a in assessment_attempt.iterrows():
            where_clause_qs = f"""
            JOIN lessons ls ON ls.id = qs.lessonId
            WHERE qs.assessmentId = '{a['assessmentId']}'
            """
            questions_raw = db.select("questions qs", "qs.*,qs.id AS question_id, ls.id, ls.content AS lesson_title, ls.slug AS lesson_slug", where_clause_qs)
            for row in questions_raw:
                questions[row['question_id']] = {
                    'question_title': row['questionText'],
                    'lessonId': row['lessonId'],
                    'point': int(row['points']),
                    'orderIndex': int(row['orderIndex']),
                    'lesson_title': row['lesson_title'],
                    'lesson_slug': row['lesson_slug'],
                    'correctAnswer': json.loads(row['correctAnswer']) if row['correctAnswer'] else [],
                    'difficulty': row['difficulty']
                }
        # print(assessmentId)
        # print("Check", question_rows)
        
        # print("||")
    
        # print(questions)
     

       
        # df.groupby(["name", "course_title"]).agg({
        #     'score': ['count', 'mean'],
        #     'percentage': 'mean',
        #     'timeTaken': 'mean'
        # }).round(2)
        
        # print(assessment_attempt)
        
        

        # assessment_attempt.columns = ['score_count', 'score_mean', 'percentage_mean', 'timeTaken_mean']

        # Dictionary để theo dõi lesson errors
        lesson_errors = {}

        for attempt in assessment_attempts:
          
            student_answers = json.loads(attempt['answers']) if attempt['answers'] else {}
            # print("xem xet", student_answers)
            course_title = attempt.get('course_title')
            category_name = attempt.get('name')
            total_math = 0
            current_math = []
            current_math.append(attempt.get('score'))
            math = 0
            total_math = attempt.get('maxScore')

            for qid, ans in student_answers.items():
                correct = questions.get(qid)
                # print(question_math)
                if not correct:
                    continue
                # question_math = correct.get('point')
                    
                lesson_id = attempt.get('lessonId') or correct.get('lessonId')
                lesson_title = correct.get('lesson_title')
                lessonslug = correct.get('lesson_slug')
                # print(correct)
             
                
                # Simplified comparison logic
                is_correct = (
                    ans == correct['correctAnswer'] if isinstance(ans, list) and isinstance(correct['correctAnswer'], list)
                    else (len(ans) == 1 and ans[0] in correct['correctAnswer']) if isinstance(ans, list)
                    else ans in correct['correctAnswer'] if isinstance(correct['correctAnswer'], list)
                    else ans == correct['correctAnswer']
                )
                
                
                # if isinstance(ans, list) and isinstance(correct['correctAnswer'], list):
                #     is_correct = ans == correct['correctAnswer']
                #     total_math = total_math + question_math
                # elif isinstance(ans, list):
                #     is_correct = len(ans) == 1 and ans[0] in correct['correctAnswer']
                #     total_math = total_math + question_math
                # elif isinstance(correct['correctAnswer'], list):
                #     is_correct = ans in correct['correctAnswer']
                #     total_math = total_math + question_math
                # else:
                #     is_correct = ans == correct['correctAnswer']
                #     total_math = total_math + question_math
                    
                    
                if is_correct:
                    math = max(current_math)
                # print("Điểm thực", current_math)    
                # print("Điểm",math)
                # print("Tổng điểm", total_math)

                
                
                
                if lesson_id:
                    lesson_key = f"{lesson_id}_{lesson_title}"
                    
                    
                    if lesson_key not in lesson_errors:
                        lesson_errors[lesson_key] = {
                            'lesson_id': lesson_id,
                            'lesson_title': lesson_title,
                            'lesson_slug': lessonslug,
                            'course_title': course_title,
                            'category_name': category_name,
                            'error_count': 0,
                            'correct_count': 0,  
                            'difficulties': set(),
                            'questions_wrong': [],
                        }
                        
                    # lesson_errors[lesson_key]['maxScore']= total_math
                    if not is_correct:
                        lesson_errors[lesson_key]['error_count'] += 1
                        lesson_errors[lesson_key]['difficulties'].add(correct['difficulty'])
                        # lesson_errors[lesson_key]['orderIndex'] = correct.get('orderIndex')
                        # lesson_errors[lesson_key]['questions_wrong'].append(correct['question_title'])
                        lesson_errors[lesson_key]['questions_wrong'].append({
                            'title': correct['question_title'],
                            'orderIndex': correct['orderIndex']
                        })
                        lesson_errors[lesson_key]['questions_wrong'].sort(key=lambda x: x['orderIndex'])
                    else:
                        lesson_errors[lesson_key]['correct_count'] += 1
                        # lesson_errors[lesson_key]['score'] += question_math
                        
                    # print(lesson_errors)
                # lesson_errors.append(total_math)
                # math = lesson_errors[lesson_key]['score']
                # print(math)
                if qid not in seen_qids:
                    question_ids.append(qid)
                    seen_qids.add(qid)

                    difficulty_stats.append({
                        'id_question': qid,
                        'difficulty': correct['difficulty'],
                        'is_correct': int(is_correct)
                    })
                
                # question_ids.append(qid)
                # difficulty_stats.append({
                #     'id_question': qid,
                #     'difficulty': correct['difficulty'],
                #     'is_correct': int(is_correct)
                # })
                
                
                
                # print(difficulty_stats)

        # Create lesson recommendations
        # print(f"🔍 Total lesson_errors found: {len(lesson_errors)}")

        priority_map = {5: 'CRITICAL', 3: 'HIGH', 2: 'MEDIUM', 1: 'LOW'}

        for lesson_data in lesson_errors.values():
            error_count = lesson_data['error_count']
            correct_count = lesson_data['correct_count']
            total_questions = error_count + correct_count
            # sorted_wrong = sorted(lesson_data['questions_wrong'], key=lambda x: x['orderIndex'])[:5]
            # lesson_data['questions_wrong'] = ' -> '.join([q['title'] for q in sorted_wrong])

            # ✅ TÍNH ACCURACY CHO LESSON
            lesson_accuracy = round((correct_count / total_questions * 100), 2) if total_questions > 0 else 0
            priority = next((p for threshold, p in sorted(priority_map.items(), reverse=True) 
                            if error_count >= threshold), 'LOW')
            # print(priority)
            
            lesson_recommendations.append({
                'lesson_id': lesson_data['lesson_id'],
                'lesson_title': lesson_data['lesson_title'],
                'lesson_slug': lesson_data['lesson_slug'],
                'course_title': lesson_data['course_title'],
                'category_name': lesson_data['category_name'],
                'error_count': error_count,
                'correct_count': correct_count, 
                'total_question_lesson': total_questions,  
                'lesson_accuracy': lesson_accuracy,  
                'priority': priority,
                'difficulties_affected': sorted(list(lesson_data['difficulties'])),
                'questions_wrong': lesson_data['questions_wrong'][:5],
                'reason': f"Bạn đã trả lời sai {error_count} câu hỏi trong bài học này"
            })

        lesson_recommendations.sort(key=lambda x: x['error_count'], reverse=True)
        # print(lesson_recommendations)

        # Process difficulty statistics
        df_difficulty = pd.DataFrame(difficulty_stats)
        if not df_difficulty.empty:
            difficulty_summary = df_difficulty.groupby("difficulty").agg(
                qid_list=('id_question', lambda x: list(x)),
                is_correct_count=('is_correct', 'count'),
                is_correct_sum=('is_correct', 'sum'),
                is_correct_mean=('is_correct', 'mean')
            ).round(2)
            # print(difficulty_summary)
            
            difficulty_summary.columns = ['id_question','total_questions', 'correct_answers', 'accuracy']
            difficulty_summary['accuracy'] = (difficulty_summary['accuracy'] * 100).round(2)
        else:
            difficulty_summary = pd.DataFrame()

        # Process categories based on performance
        total_accuracy = float(df_difficulty['is_correct'].mean()) if not df_difficulty.empty else 0
        # print(total_accuracy)
        for idx, row in assessment_attempt.iterrows():
            
            cat_name = row["name"]
            course_name = row["course_title"]
            if total_math:
                score_mean = (math/total_math) * 100
                # print(score_mean)
            else:
                continue
            # max_score = total_math
            # print(max_score)
            # print(score)
            # percentage = float(row['percentage'])
            
       
            
            # print(row_category)
            
            for difficulty, row_difficulty in difficulty_summary.iterrows():
                if total_accuracy < 100:
                    category_data = {
                        'id_user': user_id,
                        'id_assessment': row['id'],
                        'category': cat_name,
                        'course_name': course_name,
                        'correct_answers': row_difficulty['correct_answers'],
                        'accuracy_correct': row_difficulty['accuracy'],
                        'difficulty_question': difficulty,
                        'priority': 'high' if math <= 4.0 else 'medium' if math < 9 else None
                    }
                    
                    if score_mean <= 50:
                        weak_categories.append(category_data)
                    elif 50 < score_mean < 100:
                        avd_categories.append(category_data)
                        # print(category_data)
                        
            

        # # Convert MultiIndex DataFrame to JSON-serializable format
        # assessment_attempt_performance = {}
        # for idx, row in assessment_attempt.iterrows():
        #     # print(cat_name)
        #     cat_name = row["name"]
        #     course_name = row["course_title"]
        #     score = row["score"]
        #     score_count = len(score)
        #     max_score = row["maxScore"]
        #     key = f"{cat_name} - {course_name} - {lesson_title}"
        #     assessment_attempt_performance[key] = {
        #         'score_count': int(score_count),
        #         'percentage_mean': float(percentage),
        #         # 'timeTaken_mean': float(row_category['timeTaken_mean'])
        #     }

        # Fix: Convert difficulty DataFrame to JSON-serializable format
        if not difficulty_summary.empty:
            for difficulty, row_difficulty in difficulty_summary.iterrows():
                difficulty_performance[str(difficulty)] = {
                    'total_questions': int(row_difficulty['total_questions']),
                    'correct_answers': int(row_difficulty['correct_answers']),
                    'accuracy': float(row_difficulty['accuracy'])
                }
                


        # Tính accuracy tổng
      
        # total_time = sum(item.get('timeTaken_mean', 0.0) for item in assessment_attempt_performance.values())
        
        analysis = {
            'has_data': True,
            'total_questions': len(question_ids),
            'correct_answers': int(df_difficulty['is_correct'].sum()) if not df_difficulty.empty else 0,
            'overall_accuracy': total_accuracy,
            'overall_accuracy_percent': round(total_accuracy * 100, 2),
            # 'total_time': total_time,
            # 'assessment_attempt_performance': assessment_attempt_performance,
            'difficulty_performance': difficulty_performance,
            'weak_categories': weak_categories,
            'avd_categories': avd_categories,
            'lesson_recommendations': lesson_recommendations  
        }
        # print(df_difficulty['is_correct'])
        # print("check:", analysis)
        return analysis
    
  
        
        
        
        
class LearningStrategyAI:
    """AI model để quyết định chiến lược học tập"""
    
    STRATEGIES = {
        0: 'INTENSIVE_FOUNDATION',   # Học lại từ đầu
        1: 'GUIDED_PRACTICE',        # Luyện tập có hướng dẫn
        2: 'ADAPTIVE_LEARNING',      # Học thích ứng
        3: 'CHALLENGE_MODE',         # Thử thách nâng cao
        4: 'MIXED_APPROACH'          # Kết hợp nhiều phương pháp
    }
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,        # Tăng từ 100
            max_depth=10,            # Tăng từ 8
            min_samples_split=3,     # Giảm từ 5
            min_samples_leaf=2,      # Thêm
            class_weight='balanced', # Thêm để handle imbalanced data
            random_state=42,
            n_jobs=-1               # Parallel processing
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.db = DatabaseManager()
        if not self.db.connection:
            self.db.connect()
        
        # Feature names for interpretability
        # self.feature_names = [
        #     'accuracy_norm', 'total_questions_norm', 'hint_usage_rate', 'total_time_norm',
        #     'easy_accuracy', 'medium_accuracy', 'hard_accuracy',
        #     'weak_categories_norm', 'min_weak_accuracy', 'avg_weak_accuracy',
        #     'easy_time_norm', 'medium_time_norm', 'hard_time_norm'
        # ]
    
    def extract_features(self, performance_data: Dict) -> np.array:
        """Extract features từ performance analysis - FIXED VERSION"""
        
        if not performance_data.get('has_data'):
            return np.zeros(16)
        
        features = []
        
       
        features.extend([
            performance_data.get('overall_accuracy'),
            # performance_data.get('overall_accuracy_percent', 0) / 100,  # [0,1]
            performance_data.get('total_questions', 0),          # [0,1]
            0.0,  # hint_usage_rate - không có trong data, dùng default
            performance_data.get('total_time', 0) / 120               # [0,1]
        ])
        
        # 2. Difficulty-based performance (3 features) - SỬA LỖI
        diff_perf = performance_data.get('difficulty_performance', {})
        
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in diff_perf and isinstance(diff_perf[difficulty], dict):
              
                accuracy = diff_perf[difficulty].get('accuracy', 50.0) / 100
                features.append(float(accuracy))
                features.append(1.0) #có làm
            else:
                features.append(0.0)  
                features.append(0.0)
       
        weak_cats = performance_data.get('weak_categories', [])
        features.extend([
            len(weak_cats) / 5, 
            min([cat['accuracy_correct'] for cat in weak_cats]) / 100 if weak_cats else 1.0,
            np.mean([cat['accuracy_correct'] for cat in weak_cats]) / 100 if weak_cats else 1.0
        ])
        
        
        total_time = performance_data.get('total_time', 0)
        features.extend([
            min((total_time * 0.7) / 60, 1.0),   
            min((total_time * 1.0) / 90, 1.0),   
            min((total_time * 1.3) / 120, 1.0) 
        ])
        
        # 5. Learning behavior (2 features) - THÊM MỚI  
        # total_q = performance_data.get('total_questions', 1)
        # correct = performance_data.get('correct_answers', 0)
        # features.extend([
        #     (total_q - correct) / total_q if total_q > 0 else 0,  # Error rate [0,1]
        #     correct / total_q if total_q > 0 else 0               # Success rate [0,1]
        # ])
        
        
        
        
    
        # clean_features = []
        # for f in features:
        #     if isinstance(f, (int, float)):
        #         val = float(f)
        #     else:
        #         val = 0.0
        #     clean_features.append(max(0.0, min(val, 1.5)))
        # print(performance_data)
        # print(features)
        # features = clean_features
        
        # print(f"📊 Final normalized features ({len(features)}): {[round(f, 3) for f in features]}")
        
        return np.array(features[:16])
    
    
    
    def train_model(self):
        X_train = []
        y_train = []
        # db = self.db
        analysis = TestAnalyzer()
        Ai = LearningStrategyAI()
        where_clause = f""" 
        JOIN user_roles url ON url.user_id = ur.id
        JOIN roles rl ON rl.id = url.role_id
        WHERE rl.name = 'student'
        """
        students = self.db.select("users ur", " DISTINCT ur.id", where_clause)
        # print("check", student)
        for student in students:
            studentId = student['id']
            # print(studentId)
            where_clause = f"""
            WHERE studentId = '{studentId}'
            """
            assesment_attemps = self.db.select("assessment_attempts", "DISTINCT id", where_clause)
            for assesment_attemp in assesment_attemps:
                assesment_attempId = assesment_attemp['id']
                train_analysis = analysis.analyze_user_performance(self.db, studentId, assesment_attempId)
                features = Ai.extract_features(train_analysis)
                
                # print("D", features [0])
                # if features[0] < 0.4 and features[4] < 0.6:
                #     X_train.append(features [:16])
                #     y_train.append(0)
                # elif 0.4 <= features[0] < 0.65 and features[4] > 0.85 and features[6] < 0.6:
                #     X_train.append(features [:16])
                #     y_train.append(1)
                # elif 0.65 <= features[0] < 0.85 and features[10] > 0 and features[10] <= 0.4:
                #     X_train.append(features [:16])
                #     y_train.append(2)
                # elif features[0] > 0.85 and features[4] > 0.95 and features[6] > 0.9:
                #     X_train.append(features [:16])
                #     y_train.append(3)
                # else:
                #     X_train.append(features [:16])
                #     y_train.append(4)
                    
                    
                if features[0] < 0.4:
                    X_train.append(features [:16])
                    y_train.append(0)
                elif 0.4 <= features[0] < 0.65:
                    X_train.append(features [:16])
                    y_train.append(1)
                elif 0.65 <= features[0] < 0.85:
                    X_train.append(features [:16])
                    y_train.append(2)
                elif features[0] > 0.85:
                    X_train.append(features [:16])
                    y_train.append(3)
                else:
                    X_train.append(features [:16])
                    y_train.append(4)


            # print(assesments)
            
            
                # X_train.append(features[:16])
                # y_train.append(4)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Split data for evaluation
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_test_scaled = self.scaler.transform(X_test_split)
        
        
        # Thử với RandomForest thay vì GradientBoosting để có độ chính xác cao hơn
        self.model = RandomForestClassifier(
            n_estimators=200,           # Tăng số lượng trees
            max_depth=15,               # Độ sâu tối đa
            min_samples_split=5,        # Số mẫu tối thiểu để split
            min_samples_leaf=2,         # Số mẫu tối thiểu ở leaf
            max_features='sqrt',        # Số features khi split
            random_state=42,
            n_jobs=-1,                  # Sử dụng tất cả CPU cores
            class_weight='balanced'     # Cân bằng các class
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train_split)
        self.is_trained = True
        
        # Đánh giá
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n📊 Model Evaluation:")
        print(f"   Accuracy: {accuracy_score(y_test_split, y_pred):.2%}")
        # # # print("\n   Classification Report:")
        target_names = [self.STRATEGIES[i] for i in sorted(self.STRATEGIES.keys())]

        print(classification_report(
            y_test_split, 
            y_pred, 
            labels=[0, 1, 2, 3, 4],  # ép đủ 5 class
            target_names=target_names,
            digits=3,
            zero_division=0
        ))
        # print("\n   Confusion Matrix:")
        print(confusion_matrix(y_test_split, y_pred))
        
        # # Feature importance analysis
        feature_names = [
            'accuracy', 'questions', 'hints', 'time',
            'easy_acc', 'easy_flag', 'medium_acc', 'medium_flag',
            'hard_acc', 'hard_flag', 'weak_count', 'min_weak',
            'avg_weak', 'easy_time', 'medium_time', 'hard_time'
        ]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_split, 
                                    cv=5, scoring='accuracy')
        print(f"\n📈 Cross-validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
        
        return self.model
                
    # def predict_strategy(self, features: np.array) -> Tuple[str, float, Dict]:
    #     """Predict learning strategy with improved rules and model integration"""
    #     if not self.is_trained:
    #         self.train_model()
        
    #     features = np.array(features, dtype=float).flatten()
    #     accuracy = features[0]
        
    #     # Enhanced rules-based system with more conditions
    #     additional_info = {
    #         'prediction_method': 'rules_based',
    #         'reason': '',
    #         'confidence_factors': []
    #     }
        

    #     if accuracy < 0.3:
    #         additional_info['reason'] = 'Very low accuracy - requires intensive foundation building'
    #         additional_info['confidence_factors'] = ['accuracy_below_30_percent']
    #         return 'INTENSIVE_FOUNDATION', 0.95, additional_info
        
 
    #     elif accuracy >= 0.3 and accuracy < 0.5:
    #         additional_info['reason'] = 'Low-medium accuracy - guided practice recommended'
    #         additional_info['confidence_factors'] = ['accuracy_30_to_50_percent']
    #         return 'GUIDED_PRACTICE', 0.90, additional_info
        
      
    #     elif accuracy >= 0.5 and accuracy < 0.7:
    #         additional_info['reason'] = 'Medium accuracy - adaptive learning approach'
    #         additional_info['confidence_factors'] = ['accuracy_50_to_70_percent']
    #         return 'ADAPTIVE_LEARNING', 0.85, additional_info
        
        
    #     elif accuracy >= 0.7 and accuracy < 0.9:
    #         additional_info['reason'] = 'High accuracy - challenge mode appropriate'
    #         additional_info['confidence_factors'] = ['accuracy_70_to_90_percent']
    #         return 'CHALLENGE_MODE', 0.90, additional_info
        

    #     elif accuracy >= 0.9:
    #         additional_info['reason'] = 'Very high accuracy - mixed approach for continued growth'
    #         additional_info['confidence_factors'] = ['accuracy_above_90_percent']
    #         return 'MIXED_APPROACH', 0.88, additional_info
        

    #     try:
    #         features_scaled = self.scaler.transform(features.reshape(1, -1))
    #         strategy_id = self.model.predict(features_scaled)[0]
    #         probabilities = self.model.predict_proba(features_scaled)[0]
            
    #         strategy = self.STRATEGIES[strategy_id]
    #         confidence = float(probabilities[strategy_id])
            
    #         additional_info.update({
    #             'prediction_method': 'ml_model',
    #             'reason': f'ML model prediction based on feature analysis',
    #             'confidence_factors': ['ml_model_prediction'],
    #             'all_probabilities': {
    #                 self.STRATEGIES[i]: float(prob) 
    #                 for i, prob in enumerate(probabilities)
    #             }
    #         })
            
    #         return strategy, confidence, additional_info
            
    #     except Exception as e:
    #         # Final fallback
    #         additional_info.update({
    #             'prediction_method': 'fallback',
    #             'reason': f'Fallback prediction due to error: {str(e)}',
    #             'confidence_factors': ['fallback_default'],
    #             'error': str(e)
    #         })
    #         return 'ADAPTIVE_LEARNING', 0.5, additional_info
    
    
    def predict_strategy(self, features: np.array) -> Tuple[str, float, Dict]:
        if not self.is_trained:
            self.train_model()

        features = np.array(features, dtype=float).flatten()

        # Safety check cực đoan
        if features[0] < 0.1:
            return 'INTENSIVE_FOUNDATION', 0.99, {
                'prediction_method': 'safety_rule',
                'reason': 'Accuracy cực thấp (<10%)',
                'confidence_factors': ['accuracy_below_10']
            }

        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            strategy_id = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]

            strategy = self.STRATEGIES[strategy_id]
            confidence = float(probabilities[strategy_id])
            # print(f"🔮 Dự đoán: {strategy} (Độ tin cậy: {confidence:.2%})")
            return strategy, confidence, {
                'prediction_method': 'ml_model',
                'reason': strategy,
                'confidence_factors': ['ml_prediction'],
                'all_probabilities': {
                    self.STRATEGIES[i]: float(prob) for i, prob in enumerate(probabilities)
                }
            }

        except Exception as e:
            return 'ADAPTIVE_LEARNING', 0.5, {
                'prediction_method': 'fallback',
                'reason': f'Lỗi khi predict: {e}',
                'confidence_factors': ['fallback_default']
            }
    
    def save_model(self, filepath: str):
        """
        Lưu model vào file
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train. Hãy gọi train_model() trước!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'strategies': self.STRATEGIES,
            'model_type': 'LearningStrategyAI'
        }
        
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model đã được lưu tại: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model từ file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File không tồn tại: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Kiểm tra loại model
        if model_data.get('model_type') != 'LearningStrategyAI':
            raise ValueError("File không phải của LearningStrategyAI model")
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Model đã được load từ: {filepath}")
        
    @classmethod
    def load_pretrained(cls, filepath: str):
        """
        Tạo instance mới và load model
        """
        instance = cls()
        instance.load_model(filepath)
        return instance
    
class ContentRecommender:
    def recommend_lessons(self, db_manager: DatabaseManager, strategy: str, user_performance: Dict) -> List[Dict]:
        recommendations = []
        
        # Strategy configuration    
        criteria = {
            'INTENSIVE_FOUNDATION': {'levels': ['beginner'], 'focus': 'theory', 'order': 'easiest_first'},
            'GUIDED_PRACTICE': {'levels': ['beginner', 'intermediate'], 'focus': 'balanced', 'order': 'progressive'},
            'ADAPTIVE_LEARNING': {'levels': ['intermediate'], 'focus': 'practice', 'order': 'adaptive'},
            'CHALLENGE_MODE': {
                'levels': ['advanced', 'expert'], 
                'focus': 'advanced', 
                'order': 'hardest_first',
                'fallback_levels': ['intermediate', 'beginner']
            },
            'MIXED_APPROACH': {'levels': ['beginner', 'intermediate', 'advanced'], 'focus': 'mixed', 'order': 'mixed'}
        }
        
        strategy_criteria = criteria.get(strategy, criteria['MIXED_APPROACH'])
        # print(user_performance)
        
        # # Get available levels from database
                # Fix actual_levels -> danh sách (không phải tuple)
        available_courses = db_manager.select("courses", "DISTINCT level")
        available_levels = [course['level'] for course in available_courses]
        actual_levels = [level for level in strategy_criteria['levels'] if level in available_levels]
        # nếu không có level phù hợp, fallback toàn bộ available_levels (hoặc giữ strategy default)
        if not actual_levels:
            actual_levels = strategy_criteria.get('fallback_levels', strategy_criteria['levels'])

        # Normalize recomment (loại trùng theo lesson_id, giữ thứ tự lần đầu xuất hiện)
        recomment_raw = user_performance.get('lesson_recommendations', []) or []
        unique_recomment = []
        seen_lessons = set()
        for ld in recomment_raw:
            lid = ld.get('lesson_id') or ld.get('lesson_slug')  # fallback if id missing
            if not lid:
                continue
            if lid in seen_lessons:
                continue
            seen_lessons.add(lid)
            unique_recomment.append(ld)

        weak_categories = user_performance.get('weak_categories', []) or []
        avd_categories = user_performance.get('avd_categories', []) or []

        # helper: convert lesson_accuracy to number safely
        def as_number(x, default=0):
            try:
                return float(x)
            except Exception:
                return default

        recommendations = []
        added_keys = set()  # dedupe final recommendations by (lesson_id, category)

        # build function to append safely
        def try_append_recomment(lesson_data, data):
            lid = lesson_data.get('lesson_id') or lesson_data.get('lesson_slug')
            if not lid:
                return
            key = (lid, data.get('category'))
            if key in added_keys:
                return
            # optional: skip very high accuracy lessons
            if as_number(lesson_data.get('lesson_accuracy', 0)) >= 90:
                return
            recomment_dict = {
                'lesson_id': lid,
                'title': f"Ôn tập {lesson_data.get('lesson_title')}",
                'course_title': data.get('course_name'),
                'level': actual_levels,
                'category_name': data.get('category'),
                'accuracy': data.get('accuracy_correct', 0),
                'error_count': lesson_data.get('error_count', 0),
                'difficulty': data.get('difficulty_question', 'medium'),
                'priority': data.get('priority', 'MEDIUM'),
                'correct_answers': data.get('correct_answers', 0),
                'total_question_lesson': lesson_data.get('total_question_lesson', 0),
                'lesson_slug': lesson_data.get('lesson_slug', 'N/A'),
                'lesson_accuracy': as_number(lesson_data.get('lesson_accuracy', 0)),
                'difficulties_affected': lesson_data.get('difficulties_affected', []),
                'questions_wrong': lesson_data.get('questions_wrong', []),
                'recommendation_reason': lesson_data.get('reason', ''),
            }

            priority_score = self._calculate_priority_from_performance(
                recomment_dict,
                strategy_criteria,
                user_performance
            )
            recomment_dict['priority_score'] = priority_score
            recommendations.append(recomment_dict)
            added_keys.add(key)

        # 1) ưu tiên weak_categories nếu có
        if weak_categories:
            for data in weak_categories[:5]:
                for lesson_data in unique_recomment:
                    try_append_recomment(lesson_data, data)

        # 2) nếu không có weak_categories thì dùng avd_categories
        elif avd_categories:
            for data in avd_categories[:5]:
                for lesson_data in unique_recomment:
                    try_append_recomment(lesson_data, data)

        # sort & limit
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        return recommendations[:5]

        
      
        
        
    def _calculate_priority_from_performance(self, lesson_dict: Dict, 
                                           strategy_criteria: Dict, 
                                           user_performance: Dict) -> float:
        """Calculate priority score from performance data"""
        priority = 0.5  # Base score
        
        # Adjust based on error count
        error_count = lesson_dict.get('error_count', 0)
        if error_count > 0:
            priority += 0.2 * min(error_count / 3, 1.0)  # More errors = higher priority
        
        # Adjust based on priority level
        priority_level = lesson_dict.get('priority', 'MEDIUM')
        if priority_level == 'HIGH':
            priority += 0.3
        elif priority_level == 'MEDIUM':
            priority += 0.1
        
        # Adjust based on strategy
        if strategy_criteria['focus'] == 'theory' and 'theory' in lesson_dict.get('title', '').lower():
            priority += 0.2
        elif strategy_criteria['focus'] == 'practice' and 'practice' in lesson_dict.get('title', '').lower():
            priority += 0.2
        
        return min(priority, 1.0)
    
    
    
class Learning_assessment:
    def __init__(self, db):
        self.db = db
        
    def learning_analytics_data(self, user_id: str, lesson_id: str):
        now = datetime.now()
        data = {
            'user_id': user_id,
            'lesson_id': lesson_id,
            'collected_at': datetime.now().isoformat(),
            
            'session' : self._get_session(user_id, now),
            'session_analytics' : self._get_session_analytics(user_id, now),
            'total_time_for_lesson' : self._get_time_study_lesson(user_id),
            
            
        }
        
        # print(data)
        return data
    
    def _get_session_analytics(self, user_id: str, now):
        where_clause = f""" 
        WHERE lea.studentId = '{user_id}' AND MONTH(lea.date) = '{now.month} ORDER BY MONTH(lea.date) ASC'
        """
        learning_analytics = self.db.select("learning_analytics lea", "lea.totalTimeSpent, lea.readingTime, lea.videoWatchTime, lea.date", where_clause)
        days_gap_study = self._analytics_days_gap_study(learning_analytics)
        
        return {
            'learning_analytics' : learning_analytics,
            'days_study': len(learning_analytics),
            'day_gap_study': days_gap_study
        }
    
    def _get_session(self, user_id: str, now):
        where_clause = f"""
            WHERE ls.studentId = '{user_id}' AND ls.status = 'completed' AND MONTH(ls.startTime) = '{now.month}'
            ORDER BY ls.startTime ASC   
            """
        session = self.db.select("learning_sessions ls", "DISTINCT ls.startTime, ls.endTime, ls.status, ls.duration, ls.activitiesCount",where_clause)
        # session_time = self.db.select("learning_sessions ls", "SUM(ls.duration) AS total_duration_session, ROUND(AVG(ls.duration), 2) AS avg_duration_session, ROUND(STDDEV_POP(ls.duration), 2) AS std_duration_session ", where_clause)
        
        analysis_session = self._analysis_session(session)
        #Tính tổng số phiên, sự chênh lêch giữa các phiên, thời lượng trung bình giữa các phiên
        # print(session)  
        # print(analysis_session)
        # print(session[0]['endTime'])
       
        
        return{
            'session': session,
            'day_off': analysis_session
            
        }
        
  
        
        
    
    def _get_time_study_lesson(self, user_id : str):
        now = datetime.now()
        where_clause = f"JOIN lessons ls ON ls.id = lsp.lessonId WHERE lsp.studentId = '{user_id}' AND lsp.status = 'in_progress' ORDER BY lsp.lastAccessedAt DESC"
        
        lesson_progress = self.db.select("lesson_progress lsp", "DISTINCT ls.videoDuration,ls.estimatedDuration, lsp.timeSpent, lsp.progressPercentage, lsp.firstAccessedAt, lsp.lastAccessedAt, lsp.status", where_clause)
        # lesson_progress_time = self.db.select("lesson_progress lsp", "ROUND(SUM(lsp.timeSpent),2) AS total_time_study, ROUND(AVG(lsp.timeSpent),2) AS avg_time_study, ROUND(STDDEV_POP(lsp.timeSpent), 2) AS std_time_study", where_clause)
        # print(lesson_progress_time)
        
        analysis_time = self._analysis_time_(lesson_progress)
        # print(analysis_time)
        
        return{
            'time': lesson_progress,
            'learning_attitude': analysis_time
        }
        
        
    def _analysis_session(self, session): # 0,1, 2
        day_gap = []
        # print(session)
        
        for i in range(1, len(session)):
            temp = session[i-1]['endTime']
            # print(temp)
            current = session[i]['startTime'] 
            # print(current)
            gap_days = (current.date() - temp.date()).days
            # print(gap_days)
            day_gap.append(gap_days)
        #     print("t")
        #     print("|")
        #     print(temp)
        #     print("|")
        #     print(current)
            
        # print(day_gap)
            
        return day_gap
    
    def _analysis_time_(self, time_study_list: list):
        results = {}
        time_study = []
        estimatedDuration = []
        videoDuration = []
        # print(time_study_list)
        # days_study = []
        # print(time_study_list)
        # if time_study_list[0]['lastAccessedAt'] and time_study_list[0]['firstAccessedAt']:
        #     days_study = max((time_study_list[0]['lastAccessedAt'].date() - time_study_list[0]['firstAccessedAt'].date()).days, 1)
        for record in time_study_list:
        
            est_min = record.get('estimatedDuration') or 0   # phút
            vid_sec = record.get('videoDuration') or 0       # giây
            time_for_ls = record.get('timeSpent', 0)
            last = record.get('lastAccessedAt', [])
            start = record.get('firstAccessedAt', [])
            if not last:
                continue
            estimatedDuration.append(est_min * 60)  # đổi phút → giây
            videoDuration.append(vid_sec)
            time_study.append(time_for_ls)

        timmeDuration = sum(estimatedDuration) + sum(videoDuration)  # tổng giây
        totalTimeStudy = sum(time_study)
        avg_sec_per_day = round((totalTimeStudy/timmeDuration) * 100, 2)
        

        # print(days_study)
        results = {
            'percent_time_study': avg_sec_per_day,
        }
        # print(videoDuration)

        return results
    def _analytics_days_gap_study(self, learning_analytics):
        days_gap_study_list = []
        for i in range(1, len(learning_analytics)):
            pre_day = learning_analytics[i - 1]['date']
            next_day = learning_analytics[i]['date']
            # print(next_day)
            days_gap_study = (next_day - pre_day).days
            days_gap_study_list.append(days_gap_study)
            
        return days_gap_study_list
        
class RandomForestLearninAttube:
    Learning_attitude = {
        0 : 'Hard_work',
        1 : 'Distraction',
        2 : 'Lazy',
        3 : 'Give_up',
        4 : 'Cramming'
    }
    
    def __init__(self):
        
        self.model = RandomForestClassifier(
            n_estimators= 100,        # Tăng từ 100
            max_depth=8,            # Tăng từ 8
            min_samples_split=5,     # Giảm từ 5
            min_samples_leaf=2,      # Thêm
            class_weight='balanced', # Thêm để handle imbalanced data
            random_state=42,
            n_jobs=-1               # Parallel processing
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features_lesson(self, data):
        
        #Tính tổng, độ lệch chuẩn, tính trung bình cho day_off, day_study và isSkipped
        duration_list = []
        activities_list = []
        timeSpentInMonth_list = []
        # timeSpent_list = []
        now = datetime.now()
        month_now = calendar.monthrange(now.year, now.month)[1]
        session_info =  data.get('session', {})
        session = session_info.get('session')
        time_info = data.get('total_time_for_lesson', {})
        session_analytics = data.get('session_analytics')
        time_study_in_day = session_analytics.get('learning_analytics')
        days_study = session_analytics.get('days_study')
        days_gap_study = session_analytics.get('day_gap_study')
        days_off = session_info.get('day_off')
        # print(days_off)
        # time = time_info.get('time')
        # print(time)
        # print(data)
        # print("||")
        # print(session_analytics)
        # print("|||")
        # print(session)
        # print("||||")
        # print(time_info)
        # days_off = session_info.get('day_off')
        attitude = time_info.get('learning_attitude')
        percent_time_study = attitude.get('percent_time_study')
        # # duration_by_month = defaultdict(int)
        for t in time_study_in_day:
            timeSpentInMonth_list.append(t.get('totalTimeSpent'))
        # print(timeSpentInMonth_list)
        
        for s in session:
            # print(s)
            activities_list.append(s.get('activitiesCount'))
            duration_list.append(s.get('duration'))
        # print(duration_list)
        # print(activities_list)
        # print(days_off)
        
        
        
            
        pre_data_user = [
            days_study,
            round((days_study / month_now), 2),
            round(np.std(days_gap_study), 2),
            round(np.mean(days_off), 2),
            round(np.std(days_off), 2),
            round(sum(timeSpentInMonth_list), 2),
            round(np.mean(timeSpentInMonth_list),2),
            round(np.std(timeSpentInMonth_list), 2),
            round(np.mean(duration_list), 2),
            round(np.std(duration_list), 2),
            round(np.mean(activities_list), 2),
            round(np.std(activities_list), 2),
            percent_time_study
        ]
        
        # print(pre_data_user)
        
        return np.array(pre_data_user[:13])
    
    
    
    def train(self):
        X_train = []
        y_train = []
        
        np.random.seed(42)
        
        samples_per_class = 100
        """
        Features:
        1. Tổng số ngày học (sum_days_study)
        2. Số trung bình ngày học (mean_days_study)
        3. Độ lệch chuẩn khoảng cách ngày học (std_days_study) - từ gaps
        4. Trung bình khoảng cách giữa các phiên đăng nhập (mean_gap_session) - từ gaps_sessions
        5. Độ lệch chuẩn khoảng cách phiên đăng nhập (std_days_gap_off) - từ gaps_sessions
        6. Tổng thời gian học trong tháng (total_timespent)
        7. Thời gian học trung bình (mean_timespent)
        8. Độ lệch chuẩn thời gian học (std_timespent)
        9. Thời gian phiên học trung bình (mean_duration)
        10. Độ lệch chuẩn thời gian phiên (std_duration)
        11. Trung bình thao tác trên mỗi phiên (men_activities)
        12. Độ lệch chuẩn số thao tác (std_activities)
        13. Phần trăm thời gian học (percent_time_study)
        """
        
        for class_id in range(5):  # 5 classes: 0-4
            for i in range(samples_per_class):
                
                if class_id == 0:  # Hard_work
              
                    sum_days_study = np.random.randint(18, 28) 
                    mean_days_study = round(sum_days_study / 30, 2)
                    
                   
                    study_dates = np.sort(np.random.choice(range(1, 31), size=sum_days_study, replace=False))
                    gaps = np.diff(study_dates) 
                    std_days_study = round(np.std(gaps) if len(gaps) > 0 else 0, 2)
                    
                   
                    extra_login_days = np.random.randint(0, 3) 
                    extra_dates = np.random.choice([d for d in range(1, 31) if d not in study_dates], 
                                                size=min(extra_login_days, 30-sum_days_study), replace=False)
                    login_dates = np.sort(np.concatenate([study_dates, extra_dates]))
                    gaps_sessions = np.diff(login_dates)
                    mean_gap_session = round(np.mean(gaps_sessions) if len(gaps_sessions) > 0 else 1, 2)
                    std_days_gap_off = round(np.std(gaps_sessions) if len(gaps_sessions) > 0 else 0.5, 2)
                    
               
                    timespent = np.random.normal(5400, 900, size=sum_days_study)
                    timespent = np.clip(timespent, 3600, 7200).astype(int)
                    total_timespent = sum(timespent)
                    mean_timespent = round(np.mean(timespent), 2)
                    std_timespent = round(np.std(timespent), 2)
                    
                   
                    duration = timespent + np.random.randint(0, 1800, size=len(timespent))
                    mean_duration = round(np.mean(duration), 2)
                    std_duration = round(np.std(duration), 2)
                    
                    
                    activities = np.random.randint(3, 7, size=sum_days_study)
                    men_activities = round(np.mean(activities), 2)
                    std_activities = round(np.std(activities), 2)
                    
                  
                    estimated_total = 720 * sum_days_study 
                    percent_time_study = round((total_timespent / estimated_total) * 100, 2)
                    
                elif class_id == 1:  # Distraction 
                    sum_days_study = np.random.randint(10, 20)  
                    mean_days_study = round(sum_days_study / 30, 2)
                   
                    study_dates = np.sort(np.random.choice(range(1, 31), size=sum_days_study, replace=False))
                    gaps = np.diff(study_dates)
                    std_days_study = round(np.std(gaps) if len(gaps) > 0 else 0, 2)
                    
                    
                    extra_login_days = np.random.randint(5, 10)
                    extra_dates = np.random.choice([d for d in range(1, 31) if d not in study_dates], 
                                                size=min(extra_login_days, 30-sum_days_study), replace=False)
                    login_dates = np.sort(np.concatenate([study_dates, extra_dates]))
                    gaps_sessions = np.diff(login_dates)
                    mean_gap_session = round(np.mean(gaps_sessions) if len(gaps_sessions) > 0 else 1, 2)
                    std_days_gap_off = round(np.std(gaps_sessions) if len(gaps_sessions) > 0 else 1, 2)
                    
                   
                    timespent = np.random.normal(2400, 800, size=sum_days_study)
                    timespent = np.clip(timespent, 600, 4500).astype(int)
                    total_timespent = sum(timespent)
                    mean_timespent = round(np.mean(timespent), 2)
                    std_timespent = round(np.std(timespent) * 1.5, 2)  
                    
                 
                    duration = timespent * np.random.uniform(2, 4, size=len(timespent))
                    mean_duration = round(np.mean(duration), 2)
                    std_duration = round(np.std(duration), 2)
                    
                    
                    activities = np.random.randint(6, 15, size=sum_days_study)
                    men_activities = round(np.mean(activities), 2)
                    std_activities = round(np.std(activities), 2)
                    
                    estimated_total = 720 * sum_days_study
                    percent_time_study = round((total_timespent / estimated_total) * 100, 2)
                    
                elif class_id == 2:  
                    sum_days_study = np.random.randint(3, 10) 
                    mean_days_study = round(sum_days_study / 30, 2)
                    
                   
                    study_dates = np.sort(np.random.choice(range(1, 31), size=sum_days_study, replace=False))
                    gaps = np.diff(study_dates) if len(study_dates) > 1 else np.array([15])
                    std_days_study = round(np.std(gaps), 2)
                    
                 
                    extra_login_days = np.random.randint(0, 2)
                    if 30-sum_days_study > 0 and extra_login_days > 0:
                        extra_dates = np.random.choice([d for d in range(1, 31) if d not in study_dates], 
                                                    size=min(extra_login_days, 30-sum_days_study), replace=False)
                        login_dates = np.sort(np.concatenate([study_dates, extra_dates]))
                    else:
                        login_dates = study_dates
                    gaps_sessions = np.diff(login_dates) if len(login_dates) > 1 else np.array([15])
                    mean_gap_session = round(np.mean(gaps_sessions), 2)
                    std_days_gap_off = round(np.std(gaps_sessions), 2)
                    
                    
                    timespent = np.random.randint(300, 1200, size=sum_days_study)
                    total_timespent = sum(timespent)
                    mean_timespent = round(np.mean(timespent), 2)
                    std_timespent = round(np.std(timespent), 2)
                    
                    duration = timespent + np.random.randint(1000, 3000, size=len(timespent))
                    mean_duration = round(np.mean(duration), 2)
                    std_duration = round(np.std(duration), 2)
                    
                    
                    activities = np.random.randint(1, 4, size=sum_days_study)
                    men_activities = round(np.mean(activities), 2)
                    std_activities = round(np.std(activities), 2)
                    
                    estimated_total = 720 * max(sum_days_study, 5)
                    percent_time_study = round((total_timespent / estimated_total) * 100, 2)
                    
                elif class_id == 3:  # Give_up 
                    give_up_type = np.random.choice(['early', 'gradual'], p=[0.4, 0.6])
                    
                    if give_up_type == 'early':  
                        sum_days_study = np.random.randint(2, 6)
                        mean_days_study = round(sum_days_study / 30, 2)
                        
                        
                        study_dates = np.sort(np.random.choice(range(1, 11), size=sum_days_study, replace=False))
                        
                     
                        extra_login_days = np.random.randint(2, 5)
                        extra_dates = np.random.choice(range(1, 15), size=min(extra_login_days, 15-sum_days_study), replace=False)
                        login_dates = np.sort(np.unique(np.concatenate([study_dates, extra_dates])))
                        
             
                        timespent = np.array([np.random.randint(600-i*100, 1200-i*150) 
                                            for i in range(sum_days_study)])
                        timespent = np.clip(timespent, 60, 3600).astype(int)
                        activities = np.random.randint(1, 3, size=sum_days_study)
                        
                    else:  
                        sum_days_study = np.random.randint(8, 15)
                        mean_days_study = round(sum_days_study / 30, 2)
                        
                    
                        num_early = int(sum_days_study * 0.7)
                        num_late = sum_days_study - num_early
                        
                        early_dates = np.random.choice(range(1, 15), size=num_early, replace=False)
                        late_dates = np.random.choice(range(20, 31), size=num_late, replace=False)
                        study_dates = np.sort(np.concatenate([early_dates, late_dates]))
                        
                        extra_login_days = np.random.randint(3, 8)
                        extra_dates = np.random.choice(range(1, 31), size=min(extra_login_days, 31-sum_days_study), replace=False)
                        login_dates = np.sort(np.unique(np.concatenate([study_dates, extra_dates])))
                        
                      
                        early_time = np.random.randint(3000, 5400, size=num_early)
                        late_time = np.random.randint(300, 1200, size=num_late)
                        timespent = np.concatenate([early_time, late_time])
                        
                        early_act = np.random.randint(4, 7, size=num_early)
                        late_act = np.random.randint(1, 3, size=num_late)
                        activities = np.concatenate([early_act, late_act])
                    
                   
                    gaps = np.diff(study_dates) if len(study_dates) > 1 else np.array([20])
                    std_days_study = round(np.std(gaps), 2)
                    
                    gaps_sessions = np.diff(login_dates) if len(login_dates) > 1 else np.array([10])
                    mean_gap_session = round(np.mean(gaps_sessions), 2)
                    std_days_gap_off = round(np.std(gaps_sessions), 2)
                    
                    total_timespent = sum(timespent)
                    mean_timespent = round(np.mean(timespent), 2)
                    std_timespent = round(np.std(timespent), 2)
                    
                    duration = timespent * np.random.uniform(1.5, 3, size=len(timespent))
                    mean_duration = round(np.mean(duration), 2)
                    std_duration = round(np.std(duration), 2)
                    
                    men_activities = round(np.mean(activities), 2)
                    std_activities = round(np.std(activities), 2)
                    
                    estimated_total = 720 * 10
                    percent_time_study = round((total_timespent / estimated_total) * 100, 2)
                    
                else:  #Cramming
                    sum_days_study = np.random.randint(5, 10)
                    mean_days_study = round(sum_days_study / 30, 2)
                    
               
                    early_count = int(sum_days_study * 0.2)
                    late_count = sum_days_study - early_count
                    
                    early_dates = np.random.choice(range(1, 20), size=early_count, replace=False)
                    late_dates = np.random.choice(range(20, 31), size=late_count, replace=False)
                    study_dates = np.sort(np.concatenate([early_dates, late_dates]))
             
                    extra_login_days = np.random.randint(2, 5)
             
                    extra_dates = np.random.choice(range(18, 31), size=min(extra_login_days, 31-sum_days_study), replace=False)
                    login_dates = np.sort(np.unique(np.concatenate([study_dates, extra_dates])))
                    
                    gaps = np.diff(study_dates)
                    std_days_study = round(np.std(gaps) if len(gaps) > 0 else 0, 2)
                    
                    gaps_sessions = np.diff(login_dates)
                    mean_gap_session = round(np.mean(gaps_sessions) if len(gaps_sessions) > 0 else 1, 2)
                    std_days_gap_off = round(np.std(gaps_sessions) * 2, 2)  # High variance
                    
             
                    early_time = np.random.randint(1800, 3600, size=early_count)
                    late_time = np.random.randint(10800, 21600, size=late_count)  # 3-6 giờ
                    timespent = np.concatenate([early_time, late_time])
                    
                    total_timespent = sum(timespent)
                    mean_timespent = round(np.mean(timespent), 2)
                    std_timespent = round(np.std(timespent), 2)
                    
                    duration = timespent + np.random.randint(0, 1800, size=len(timespent))
                    mean_duration = round(np.mean(duration), 2)
                    std_duration = round(np.std(duration), 2)
                    
                    early_act = np.random.randint(3, 5, size=early_count)
                    late_act = np.random.randint(10, 20, size=late_count)
                    activities = np.concatenate([early_act, late_act])
                    men_activities = round(np.mean(activities), 2)
                    std_activities = round(np.std(activities), 2)
                    
                    estimated_total = 720 * sum_days_study
                    percent_time_study = round((total_timespent / estimated_total) * 100, 2)
                
                
                features = [
                    sum_days_study,           # 1. Tổng số ngày học
                    mean_days_study,          # 2. Trung bình ngày học
                    std_days_study,           # 3. Std của gaps ( khoảng cách ngày học)
                    mean_gap_session,         # 4. Mean của gaps_sessions (khoảng cách ngày đăng nhập)
                    std_days_gap_off,         # 5. Std của gaps_sessions
                    total_timespent,          # 6. Tổng thời gian học
                    mean_timespent,           # 7. Thời gian học trung bình
                    std_timespent,            # 8. Std thời gian học
                    mean_duration,            # 9. Duration trung bình
                    std_duration,             # 10. Std duration
                    men_activities,           # 11. Activities trung bình
                    std_activities,           # 12. Std activities
                    percent_time_study        # 13. Phần trăm thời gian học
                ]
                
                # Add small noise
                features[5] *= np.random.uniform(0.95, 1.05)
                features[10] *= np.random.uniform(0.9, 1.1)
                
                X_train.append(features)
                y_train.append(class_id)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        

        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_val_scaled = self.scaler.transform(X_val)
        
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train_split)
        self.model = grid_search.best_estimator_
        
        # Evaluate
        # y_pred = self.model.predict(X_val_scaled)
        
        labels = sorted(set(y_val))
        # target_names = [self.Learning_attitude[i] for i in labels]
        
        # print("Best parameters:", grid_search.best_params_)
        # print("\nClassification Report:")
        # print(classification_report(
        #     y_val, y_pred, target_names=target_names, digits=2
        # ))
        
        # feature_names = [
        #     'sum_days_study', 'mean_days_study', 'std_days_study',
        #     'mean_gap_session', 'std_days_gap_off', 'total_timespent',
        #     'mean_timespent', 'std_timespent', 'mean_duration',
        #     'std_duration', 'men_activities', 'std_activities',
        #     'percent_time_study'
        # ]
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # print("\nFeature Importance:")
        # for i in range(len(feature_names)):
        #     print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        self.is_trained = True
        return X_train, y_train

    # Helper function để tính percent_time_study cho real data
    def calculate_percent_time_study(total_time_spent, estimated_duration_minutes, video_duration_seconds=None):
        """
        Calculate percent_time_study consistently
        
        Args:
            total_time_spent: Total time spent in seconds
            estimated_duration_minutes: Estimated duration in minutes
            video_duration_seconds: Video duration in seconds (optional)
        
        Returns:
            percent_time_study as a percentage
        """
        # Convert estimated duration to seconds
        estimated_total = estimated_duration_minutes * 60
        
        # Add video duration if available
        if video_duration_seconds:
            estimated_total += video_duration_seconds
        
        # Avoid division by zero
        if estimated_total == 0:
            return 0
        
        # Calculate percentage
        return round((total_time_spent / estimated_total) * 100, 2)
    
    
    
    def predict(self, data, return_proba=False):
        # """
            
        # Returns:
        # --------
        # dict : Dictionary chứa kết quả dự đoán
        #     - 'prediction': Class ID được dự đoán
        #     - 'attitude': Tên learning attitude
        #     - 'confidence': Độ tin cậy của dự đoán
        #     - 'probabilities': Xác suất cho mỗi class (nếu return_proba=True)
        # """
        
        # # Kiểm tra model đã được train chưa
        if not self.is_trained:
            raise ValueError("Model chưa được train. Hãy gọi train() trước!")
        
        # # Extract features nếu input là dict
        if isinstance(data, dict):
            features = self.extract_features_lesson(data)
        elif isinstance(data, (np.ndarray, list)):
            features = np.array(data)
            if features.shape[-1] != 8:
                raise ValueError(f"Expected 8 features, got {features.shape[-1]}")
        else:
            raise TypeError("Data must be dict or numpy array")
        
        # # Reshape nếu cần
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # # Scale features
        features_scaled = self.scaler.transform(features)
        
        # # Predict
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        
        # # Tạo kết quả
        result = {
            'prediction': int(prediction),
            'attitude': self.Learning_attitude[prediction],
            'confidence': float(np.max(proba)),
        }
        
        if return_proba:
            result['probabilities'] = {
                self.Learning_attitude[i]: float(prob) 
                for i, prob in enumerate(proba)
            }
        
        
        return result
    
    def save_model(self, filepath: str):
        """
        Lưu RandomForestLearninAttube model vào file
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train. Hãy gọi train() trước!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'learning_attitude': self.Learning_attitude,
            'model_type': 'RandomForestLearninAttube'
        }
        
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        joblib.dump(model_data, filepath)
        print(f"✅ RandomForestLearninAttube model đã được lưu tại: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load RandomForestLearninAttube model từ file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File không tồn tại: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Kiểm tra loại model
        if model_data.get('model_type') != 'RandomForestLearninAttube':
            raise ValueError("File không phải của RandomForestLearninAttube model")
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ RandomForestLearninAttube model đã được load từ: {filepath}")
        
    @classmethod
    def load_pretrained(cls, filepath: str):
        """
        Tạo instance mới và load model
        """
        instance = cls()
        instance.load_model(filepath)
        return instance
    
class ModelManager:
    """
    Quản lý việc lưu và load tất cả models trong hệ thống
    """
    
    @staticmethod
    def save_all_models(learning_strategy_ai: LearningStrategyAI, 
                    #    random_forest_attitude: RandomForestLearninAttube,
                       base_path: str = "./models/"):
        """
        Lưu tất cả models vào thư mục
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        
        # Save LearningStrategyAI
        if learning_strategy_ai.is_trained:
            strategy_path = os.path.join(base_path, "learning_strategy_ai.joblib")
            learning_strategy_ai.save_model(strategy_path)
        else:
            print("⚠️ LearningStrategyAI chưa được train, skip save")
            
        # # Save RandomForestLearninAttube  
        # if random_forest_attitude.is_trained:
        #     attitude_path = os.path.join(base_path, "random_forest_attitude.joblib")
        #     random_forest_attitude.save_model(attitude_path)
        # else:
        #     print("⚠️ RandomForestLearninAttube chưa được train, skip save")
            
        # print(f"\n🎯 Tất cả models đã được lưu trong: {base_path}")
    
    @staticmethod
    def load_all_models(base_path: str = "./models/") -> tuple:
        """
        Load tất cả models từ thư mục
        
        Returns:
            tuple: (LearningStrategyAI, RandomForestLearninAttube)
        """
        strategy_path = os.path.join(base_path, "learning_strategy_ai.joblib")
        attitude_path = os.path.join(base_path, "random_forest_attitude.joblib")
        
        learning_strategy_ai = None
        random_forest_attitude = None
        
        # Load LearningStrategyAI
        if os.path.exists(strategy_path):
            learning_strategy_ai = LearningStrategyAI.load_pretrained(strategy_path)
        else:
            print(f"⚠️ Không tìm thấy file: {strategy_path}")
            learning_strategy_ai = LearningStrategyAI()
            
        # Load RandomForestLearninAttube
        if os.path.exists(attitude_path):
            random_forest_attitude = RandomForestLearninAttube.load_pretrained(attitude_path)
        else:
            print(f"⚠️ Không tìm thấy file: {attitude_path}")
            random_forest_attitude = RandomForestLearninAttube()
            
        print(f"🔄 Models đã được load từ: {base_path}")
        return learning_strategy_ai, random_forest_attitude
    
    @staticmethod
    def create_model_info(base_path: str = "./models/") -> dict:
        """
        Tạo thông tin về các models đã lưu
        """
        info = {
            'base_path': base_path,
            'models': {},
            'created_at': datetime.now().isoformat()
        }
        
        # Check LearningStrategyAI
        strategy_path = os.path.join(base_path, "learning_strategy_ai.joblib")
        if os.path.exists(strategy_path):
            stat = os.stat(strategy_path)
            info['models']['learning_strategy_ai'] = {
                'file_path': strategy_path,
                'file_size_mb': round(stat.st_size / 1024 / 1024, 2),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'exists': True
            }
        else:
            info['models']['learning_strategy_ai'] = {'exists': False}
            
        # Check RandomForestLearninAttube
        attitude_path = os.path.join(base_path, "random_forest_attitude.joblib")
        if os.path.exists(attitude_path):
            stat = os.stat(attitude_path)
            info['models']['random_forest_attitude'] = {
                'file_path': attitude_path,
                'file_size_mb': round(stat.st_size / 1024 / 1024, 2),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'exists': True
            }
        else:
            info['models']['random_forest_attitude'] = {'exists': False}
            
        return info


    
    
            
class AITrackingDataCollector:
    def __init__(self, db):
        self.db = db
    
    def collect_comprehensive_data(self, user_id: str, course_id: str = None):
        """
        Thu thập tất cả dữ liệu cần thiết cho AI tracking
        """
        data = {
            'user_id': user_id,
            'course_id': course_id,
            'collected_at': datetime.now().isoformat(),
            
            # 1. BASIC PROGRESS DATA
            'basic_progress': self._get_basic_progress(user_id, course_id),
            
            # 2. LEARNING ACTIVITIES - Dữ liệu hành vi chi tiết
            'learning_activities': self._get_learning_activities(user_id, course_id),
            
            # 3. LEARNING SESSIONS - Phiên học tập
            # 'learning_sessions': self._get_learning_sessions(user_id),
            
            # 4. ASSESSMENT PERFORMANCE - Kết quả kiểm tra
            'assessment_performance': self._get_assessment_performance(user_id, course_id),
            
            # 5. TIME PATTERNS - Mô hình thời gian học
            # 'time_patterns': self._get_time_patterns(user_id),
            
            # 6. ENGAGEMENT METRICS - Chỉ số tương tác
            # 'engagement_metrics': self._get_engagement_metrics(user_id, course_id),
            
            # 7. LEARNING STYLE - Phong cách học tập
            # 'learning_style': self._get_learning_style(user_id),
            
            # 8. CONTENT INTERACTION - Tương tác với nội dung
            'content_interaction': self._get_content_interaction(user_id, course_id),
            
            # 9. SOCIAL INTERACTION - Tương tác xã hội
            # 'social_interaction': self._get_social_interaction(user_id),
            
            # 10. HISTORICAL ANALYTICS - Dữ liệu phân tích lịch sử
            # 'historical_analytics': self._get_historical_analytics(user_id)
        }
        # print(data)
        return data
    def _get_basic_progress(self, user_id, course_id):
        where_clause= f""" 
        JOIN student_profiles stdpr ON stdpr.userId = '{user_id}'
        WHERE cr.id = '{course_id}' AND cr.status = 'published' AND stdpr.difficultyPreference = cr.level
        """
      
        basic = self.db.select("courses cr","cr.title, cr.slug, cr.totalLessons, cr.totalVideoDuration, cr.level, cr.durationHours, stdpr.preferredLearningStyle", where_clause)
        # print("Here", basic)
        return basic
    
    def _get_learning_activities(self, user_id, course_id):
        where_clause = f""" 
        WHERE studentId = '{user_id}' AND courseId = '{course_id}'
        """
        activities = self.db.select("learning_activities","*", where_clause)
        # print("Check:", self._analysis_activities(activities))
        return self._analysis_activities(activities)
    
    def _get_assessment_performance(self, user_id, course_id):
        """Thu thập và phân tích dữ liệu hiệu suất đánh giá của sinh viên"""
        

        where_clause = f""" 
        JOIN assessments a ON ast.assessmentId = a.id 
        LEFT JOIN lessons ls ON a.lessonId = ls.id
        LEFT JOIN courses cr ON cr.id = COALESCE(a.courseId, ls.courseId)
        WHERE ast.studentId = '{user_id}' AND (a.courseId = '{course_id}' OR ls.courseId = '{course_id}')
        AND ast.status IN ('submitted', 'graded')
        """
        
        attempts = self.db.select(
            "assessment_attempts ast",
            """ast.*, a.title, a.assessmentType, a.totalPoints, a.passingScore, 
               a.timeLimit, a.maxAttempts, ls.title AS lesson_title, cr.title AS course_title""",
            where_clause
        )
        # print(self._analyze_assessment_performance(attempts, user_id, course_id))
        return self._analyze_assessment_performance(attempts, user_id, course_id)
    
    
    def _get_content_interaction(self, user_id, course_id):
        """Thu thập và phân tích dữ liệu tương tác với nội dung của sinh viên"""
        
        interaction_data = {
            'lesson_progress': self._get_lesson_interactions(user_id, course_id),
            'video_interactions': self._get_video_interactions(user_id, course_id),
            'content_activities': self._get_content_activities(user_id, course_id),
            'notes_bookmarks': self._get_notes_bookmarks(user_id, course_id)
        }
        # print( self._analyze_comprehensive_content_interaction(interaction_data, user_id, course_id))
        return self._analyze_comprehensive_content_interaction(interaction_data, user_id, course_id)
    def _analysis_activities(self, activities):
        if not activities:
            return {
                'total_activities': 0,
                'activity_summary': {},
                'engagement_patterns': {},
                'time_analysis': {},
                'learning_behavior': {}
            }
        
       
        df = pd.DataFrame(activities)
        
        
        activity_summary = self._analyze_activity_summary(df)
        
        
        engagement_patterns = self._analyze_engagement_patterns(df)
        
      
        time_analysis = self._analyze_time_patterns(df)
        
      
        learning_behavior = self._analyze_learning_behavior(df)
        
        return {
            'total_activities': len(activities),
            'activity_summary': activity_summary,
            'engagement_patterns': engagement_patterns,
            'time_analysis': time_analysis,
            'learning_behavior': learning_behavior
        }
    
    def _analyze_activity_summary(self, df):
        """Phân tích tổng quan các loại hoạt động"""
        activity_counts = df.groupby('activityType').agg({
            'id': 'count',
            'duration': ['sum', 'mean', 'std']
        }).round(2)
        
        # print("xem 1:", activity_counts)
        
        activity_counts.columns = ['count', 'total_duration', 'avg_duration', 'std_duration']
        
       
        total_activities = len(df)
        # print(total_activities)
        activity_counts['percentage'] = (activity_counts['count'] / total_activities * 100).round(2)
        
       
        summary = {}
        for activity_type in activity_counts.index:
            row = activity_counts.loc[activity_type]
            summary[activity_type] = {
                'count': int(row['count']),
                'percentage': float(row['percentage']),
                'total_duration': float(row['total_duration']) if pd.notna(row['total_duration']) else 0,
                'avg_duration': float(row['avg_duration']) if pd.notna(row['avg_duration']) else 0,
                'std_duration': float(row['std_duration']) if pd.notna(row['std_duration']) else 0
            }
        
        return summary
    
    def _analyze_engagement_patterns(self, df):
        """Phân tích mô hình tham gia học tập"""
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        # print(df['timestamp'])
        
        
        hourly_activity = df.groupby('hour').size().to_dict()
        # print(hourly_activity)
        
       
        weekly_activity = df.groupby('day_of_week').size().to_dict()
        # print(weekly_activity)
      
        unique_dates = sorted(df['date'].unique())
        # print(unique_dates)
        streak_info = self._calculate_learning_streak(unique_dates)
        # print(streak_info)
        
        session_analysis = self._analyze_sessions(df)
        # print(session_analysis)
        
        return {
            'hourly_distribution': hourly_activity,
            'weekly_distribution': weekly_activity,
            'learning_streak': streak_info,
            'session_analysis': session_analysis,
            'most_active_hour': max(hourly_activity, key=hourly_activity.get) if hourly_activity else None,
            'most_active_day': max(weekly_activity, key=weekly_activity.get) if weekly_activity else None
        }
    
    def _analyze_time_patterns(self, df):
        """Phân tích các mô hình thời gian"""
        
        duration_df = df[df['duration'].notna() & (df['duration'] > 0)]
        
        if duration_df.empty:
            return {
                'total_learning_time': 0,
                'average_session_duration': 0,
                'time_distribution': {},
                'peak_learning_periods': []
            }
        
        total_time = duration_df['duration'].sum()
        # print("xem 2:", total_time)
        avg_duration = duration_df['duration'].mean()
        

        time_by_activity = duration_df.groupby('activityType')['duration'].agg(['sum', 'mean', 'count']).round(2)
        time_distribution = {}
        for activity in time_by_activity.index:
            time_distribution[activity] = {
                'total_time': float(time_by_activity.loc[activity, 'sum']),
                'avg_time': float(time_by_activity.loc[activity, 'mean']),
                'frequency': int(time_by_activity.loc[activity, 'count'])
            }
        # print(time_distribution)
        # Tìm các khoảng thời gian học tập cao điểm
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        hourly_time = duration_df.groupby(duration_df['timestamp'].dt.hour)['duration'].sum()
        # print(hourly_time)
        peak_hours = hourly_time.nlargest(3).index.tolist()
        # print(peak_hours)
        
        return {
            'total_learning_time': float(total_time),
            'average_session_duration': float(avg_duration),
            'time_distribution': time_distribution,
            'peak_learning_hours': peak_hours
        }
    
    def _analyze_learning_behavior(self, df):
        """Phân tích hành vi học tập"""
        # Phân tích độ tập trung (dựa trên video activities)
        video_activities = df[df['activityType'].str.contains('video', case=False, na=False)]
        # print("Here", video_activities)
        focus_analysis = self._analyze_focus_patterns(video_activities)
        
        # Phân tích mô hình hoàn thành
        completion_analysis = self._analyze_completion_patterns(df)
        
        # Phân tích tương tác với nội dung
        interaction_analysis = self._analyze_content_interaction(df)
        
        return {
            'focus_patterns': focus_analysis,
            'completion_patterns': completion_analysis,
            'content_interaction': interaction_analysis
        }
    
    def _calculate_learning_streak(self, dates):
        """Tính chuỗi ngày học liên tiếp"""
        if not dates:
            return {'current_streak': 0, 'longest_streak': 0}
        
        dates = [pd.to_datetime(date) for date in dates]
        # print("1", dates)
        dates.sort()
        
        current_streak = 1
        longest_streak = 1
        temp_streak = 1
        
        for i in range(1, len(dates)):
            diff = (dates[i] - dates[i-1]).days
            # print("1", diff)
            if diff == 1:
                temp_streak += 1
                longest_streak = max(longest_streak, temp_streak)
            else:
                temp_streak = 1
        
        # Tính current streak (từ ngày gần nhất về trước)
        today = datetime.now().date()
        last_date = dates[-1].date()
        
        if (today - last_date).days <= 1: 
            for i in range(len(dates)-1, 0, -1):
                if (dates[i] - dates[i-1]).days == 1:
                    current_streak += 1
                else:
                    break
        else:
            current_streak = 0
        
        return {
            'current_streak': current_streak,
            'longest_streak': longest_streak,
            'total_learning_days': len(dates)
        }
    
    def _analyze_sessions(self, df):
        """Phân tích các phiên học"""
        session_stats = df.groupby('sessionId').agg({
            'id': 'count',
            'duration': 'sum',
            'timestamp': ['min', 'max']
        }).round(2)
        
        session_stats.columns = ['activity_count', 'total_duration', 'start_time', 'end_time']
        
        
        session_stats['session_length'] = (
            pd.to_datetime(session_stats['end_time']) - 
            pd.to_datetime(session_stats['start_time'])
        ).dt.total_seconds()
        
        # print( pd.to_datetime(session_stats['end_time']))
        # print(pd.to_datetime(session_stats['start_time']))
        
        return {
            'total_sessions': len(session_stats),
            'avg_activities_per_session': float(session_stats['activity_count'].mean()),
            'avg_session_duration': float(session_stats['total_duration'].mean()) if session_stats['total_duration'].notna().any() else 0,
            'avg_session_length': float(session_stats['session_length'].mean())
        }
    
    def _analyze_focus_patterns(self, video_df):
        """Phân tích mô hình tập trung qua video activities"""
        if video_df.empty:
            return {'focus_score': 0, 'video_engagement': {}}
        
        pause_count = len(video_df[video_df['activityType'] == 'video_pause'])
        seek_count = len(video_df[video_df['activityType'] == 'video_seek'])
        play_count = len(video_df[video_df['activityType'] == 'video_play'])
        complete_count = len(video_df[video_df['activityType'] == 'video_complete'])
        
        
        total_video_actions = len(video_df)
        if total_video_actions > 0:
            focus_score = max(0, 100 - (pause_count + seek_count) / total_video_actions * 100)
        else:
            focus_score = 0
        
        return {
            'focus_score': round(focus_score, 2),
            'video_engagement': {
                'play_count': play_count,
                'pause_count': pause_count,
                'seek_count': seek_count,
                'complete_count': complete_count,
                'completion_rate': round(complete_count / max(play_count, 1) * 100, 2)
            }
        }
    
    def _analyze_completion_patterns(self, df):
        """Phân tích mô hình hoàn thành"""
        completion_activities = df[df['activityType'].str.contains('complete', case=False, na=False)]
        start_activities = df[df['activityType'].str.contains('start', case=False, na=False)]
        
        completion_rate = 0
        if len(start_activities) > 0:
            completion_rate = len(completion_activities) / len(start_activities) * 100
        
        return {
            'completion_rate': round(completion_rate, 2),
            'total_completions': len(completion_activities),
            'total_starts': len(start_activities)
        }
    
    def _analyze_content_interaction(self, df):
        """Phân tích tương tác với nội dung"""
        interactive_activities = df[df['activityType'].isin([
            'discussion_post', 'chat_message', 'note_create', 
            'bookmark_add', 'help_request', 'forum_post'
        ])]
        
        return {
            'interaction_count': len(interactive_activities),
            'interaction_types': interactive_activities['activityType'].value_counts().to_dict() if not interactive_activities.empty else {}
        }
    
    
    def _analyze_assessment_performance(self, attempts, user_id, course_id):
        """
        Phân tích hiệu suất đánh giá chi tiết
        
        Args:
            attempts: Danh sách các lần thử assessment
            user_id: ID sinh viên
            course_id: ID khóa học
            
        Returns:
            dict: Kết quả phân tích hiệu suất đánh giá
        """
        if not attempts:
            return {
                'total_assessments': 0,
                'assessment_summary': {},
                'performance_trends': {},
                'difficulty_analysis': {},
                'time_analysis': {},
                'improvement_analysis': {}
            }
            
        

        df = pd.DataFrame(attempts)
        

        assessment_summary = self._analyze_assessment_summary(df)
        

        performance_trends = self._analyze_performance_trends(df)
        
      
        difficulty_analysis = self._analyze_difficulty_performance(df, user_id)
        
        
        time_analysis = self._analyze_assessment_time(df)
        
        
        improvement_analysis = self._analyze_improvement_patterns(df)
        
        return {
            'total_assessments': len(attempts),
            'assessment_summary': assessment_summary,
            'performance_trends': performance_trends,
            'difficulty_analysis': difficulty_analysis,
            'time_analysis': time_analysis,
            'improvement_analysis': improvement_analysis
        }
    
    def _analyze_assessment_summary(self, df):
        """Phân tích tổng quan kết quả đánh giá"""
        
        total_attempts = len(df)
        avg_score = df['score'].mean() if 'score' in df.columns else 0
        avg_percentage = df['percentage'].mean() if 'percentage' in df.columns else 0
        
       
        passed_attempts = 0
        failed_attempts = 0
        
        for _, row in df.iterrows():
            percentage = row.get('percentage', 0)
            passing_score = row.get('passingScore', 70)
            
            if percentage is not None and percentage >= passing_score:
                passed_attempts += 1
            else:
                failed_attempts += 1
        
        
        assessment_by_type = df.groupby('assessmentType').agg({
            'score': ['count', 'mean', 'std'],
            'percentage': ['mean', 'std'],
            'timeTaken': 'mean'
        }).round(2)
        
      
        type_analysis = {}
        if not assessment_by_type.empty:
            for assessment_type in assessment_by_type.index:
                type_analysis[assessment_type] = {
                    'count': int(assessment_by_type.loc[assessment_type, ('score', 'count')]),
                    'avg_score': float(assessment_by_type.loc[assessment_type, ('score', 'mean')]) if pd.notna(assessment_by_type.loc[assessment_type, ('score', 'mean')]) else 0,
                    'avg_percentage': float(assessment_by_type.loc[assessment_type, ('percentage', 'mean')]) if pd.notna(assessment_by_type.loc[assessment_type, ('percentage', 'mean')]) else 0,
                    'avg_time': float(assessment_by_type.loc[assessment_type, ('timeTaken', 'mean')]) if pd.notna(assessment_by_type.loc[assessment_type, ('timeTaken', 'mean')]) else 0
                }
        
        return {
            'total_attempts': total_attempts,
            'average_score': float(avg_score) if pd.notna(avg_score) else 0,
            'average_percentage': float(avg_percentage) if pd.notna(avg_percentage) else 0,
            'passed_attempts': passed_attempts,
            'failed_attempts': failed_attempts,
            'pass_rate': round(passed_attempts / total_attempts * 100, 2) if total_attempts > 0 else 0,
            'assessment_by_type': type_analysis
        }
    
    def _analyze_performance_trends(self, df):
        """Phân tích xu hướng hiệu suất theo thời gian"""
        
        df_sorted = df.sort_values('submittedAt')
        
        if len(df_sorted) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'recent_performance': {}}
        
       
        scores = df_sorted['percentage'].dropna()
        if len(scores) >= 2:
            # Tính hệ số góc đơn giản
            # x = range(len(scores))
            slope = (scores.iloc[-1] - scores.iloc[0]) / (len(scores) - 1)
            # print(scores)
            if slope > 2:
                trend = 'improving'
            elif slope < -2:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
            slope = 0
        
        # Phân tích hiệu suất gần đây (3 lần gần nhất)
        recent_attempts = df_sorted.tail(3)
        recent_performance = {
            'recent_average': float(recent_attempts['percentage'].mean()) if not recent_attempts.empty else 0,
            'recent_best': float(recent_attempts['percentage'].max()) if not recent_attempts.empty else 0,
            'recent_worst': float(recent_attempts['percentage'].min()) if not recent_attempts.empty else 0
        }
        
        return {
            'trend': trend,
            'slope': round(slope, 2),
            'recent_performance': recent_performance,
            'improvement_rate': slope
        }
    
    def _analyze_difficulty_performance(self, df, user_id):
        """Phân tích hiệu suất theo độ khó câu hỏi"""
        
        difficulty_stats = {}
        
        for _, attempt in df.iterrows():
            assessment_id = attempt['assessmentId']
            
          
            questions = self.db.select(
                "questions", 
                "difficulty, points",
                f"WHERE assessmentId = '{assessment_id}'"
            )
            
            if questions:
                student_answers = json.loads(attempt.get('answers', '{}')) if attempt.get('answers') else {}
                
                for question in questions:
                    difficulty = question['difficulty']
                    if difficulty not in difficulty_stats:
                        difficulty_stats[difficulty] = {
                            'total_questions': 0,
                            'correct_answers': 0,
                            'total_points': 0,
                            'earned_points': 0
                        }
                    
                    difficulty_stats[difficulty]['total_questions'] += 1
                    difficulty_stats[difficulty]['total_points'] += question['points']
                    
        for difficulty in difficulty_stats:
            stats = difficulty_stats[difficulty]
            if stats['total_questions'] > 0:
                stats['accuracy'] = round(stats['correct_answers'] / stats['total_questions'] * 100, 2)
            else:
                stats['accuracy'] = 0
        
        return difficulty_stats
    
    def _analyze_assessment_time(self, df):
        """Phân tích thời gian làm bài"""
        
        time_data = df[df['timeTaken'].notna() & (df['timeTaken'] > 0)]
        
        if time_data.empty:
            return {
                'average_time': 0,
                'time_efficiency': {},
                'time_vs_performance': {}
            }
        
        avg_time = time_data['timeTaken'].mean()
        
        # Phân tích hiệu suất thời gian
        time_efficiency = {}
        
        # Nhóm theo thời gian (nhanh, trung bình, chậm)
        time_percentiles = time_data['timeTaken'].quantile([0.33, 0.67])
        
        for _, row in time_data.iterrows():
            time_taken = row['timeTaken']
            percentage = row.get('percentage', 0)
            
            if time_taken <= time_percentiles.iloc[0]:
                category = 'fast'
            elif time_taken <= time_percentiles.iloc[1]:
                category = 'medium'
            else:
                category = 'slow'
            
            if category not in time_efficiency:
                time_efficiency[category] = {'times': [], 'scores': []}
            time_efficiency[category]['times'].append(time_taken)
            time_efficiency[category]['scores'].append(percentage)
        
        # Tính trung bình cho mỗi category
        for category in time_efficiency:
            times = time_efficiency[category]['times']
            scores = time_efficiency[category]['scores']
            
            time_efficiency[category] = {
                'count': len(times),
                'avg_time': round(np.mean(times), 2),
                'avg_score': round(np.mean(scores), 2)
            }
        
        # Phân tích mối quan hệ thời gian vs hiệu suất
        correlation = 0
        if len(time_data) > 1:
            try:
                # Lọc dữ liệu hợp lệ
                valid_time = time_data['timeTaken'].dropna()
                valid_percentage = time_data['percentage'].dropna()
                
                # Đảm bảo cùng index để tính correlation chính xác
                common_index = valid_time.index.intersection(valid_percentage.index)
                if len(common_index) >= 2:
                    time_values = valid_time.loc[common_index]
                    percentage_values = valid_percentage.loc[common_index]
                    
                    # Kiểm tra xem có variance không (tránh correlation với constant values)
                    if time_values.std() > 0 and percentage_values.std() > 0:
                        correlation = np.corrcoef(time_values, percentage_values)[0, 1]
                        if np.isnan(correlation) or np.isinf(correlation):
                            correlation = 0
            except (ValueError, IndexError, TypeError, KeyError):
                correlation = 0
        
        return {
            'average_time': round(avg_time, 2),
            'time_efficiency': time_efficiency,
            'time_vs_performance': {
                'correlation': round(correlation, 3),
                'interpretation': self._interpret_time_correlation(correlation)
            }
        }
    
    def _analyze_improvement_patterns(self, df):
        """Phân tích mô hình cải thiện"""
        
        if len(df) < 2:
            return {'pattern': 'insufficient_data'}
        
        # Sắp xếp theo thời gian
        df_sorted = df.sort_values('submittedAt')
        
        # Tính sự thay đổi giữa các lần thử
        improvements = []
        for i in range(1, len(df_sorted)):
            prev_score = df_sorted.iloc[i-1]['percentage']
            curr_score = df_sorted.iloc[i]['percentage']
            improvement = curr_score - prev_score
            improvements.append(improvement)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            consistency = np.std(improvements)
            
            # Xác định pattern
            if avg_improvement > 5:
                pattern = 'consistent_improvement'
            elif avg_improvement < -5:
                pattern = 'declining'
            elif consistency < 10:
                pattern = 'stable'
            else:
                pattern = 'inconsistent'
        else:
            pattern = 'no_data'
            avg_improvement = 0
            consistency = 0
        
        return {
            'pattern': pattern,
            'average_improvement': round(avg_improvement, 2),
            'consistency_score': round(100 - consistency, 2),  # Càng cao càng nhất quán
            'total_improvement': round(improvements[-1] if improvements else 0, 2)
        }
    
    def _interpret_time_correlation(self, correlation):
        """Giải thích mối quan hệ giữa thời gian và hiệu suất"""
        
        if correlation > 0.3:
            return "Longer time tends to improve performance"
        elif correlation < -0.3:
            return "Longer time may indicate struggles"
        else:
            return "No clear relationship between time and performance"
    
    
    def _get_lesson_interactions(self, user_id, course_id):
        """Lấy dữ liệu tương tác với bài học"""
        
        where_clause = f"""
        JOIN lessons ls ON lp.lessonId = ls.id
        WHERE lp.studentId = '{user_id}' AND ls.courseId = '{course_id}'
        """
        
        lesson_progress = self.db.select(
            "lesson_progress lp",
            """lp.*, ls.title, ls.lessonType, ls.videoDuration, ls.estimatedDuration,
               ls.orderIndex""",
            where_clause
        )
        
        return lesson_progress
    
    def _get_video_interactions(self, user_id, course_id):
        """Lấy dữ liệu tương tác với video"""
        
        where_clause = f"""
        JOIN lessons ls ON ls.id = la.lessonId
        WHERE la.studentId = '{user_id}' AND ls.courseId = '{course_id}'
        AND la.activityType IN ('video_play', 'video_pause', 'video_seek', 'video_complete')
        """
        
        video_activities = self.db.select(
            "learning_activities la",
            "la.*, ls.title, ls.videoDuration",
            where_clause
        )
        
        return video_activities
    
    def _get_content_activities(self, user_id, course_id):
        """Lấy các hoạt động tương tác với nội dung"""
        
        where_clause = f"""
        LEFT JOIN lessons ls ON ls.id = la.lessonId
        WHERE la.studentId = '{user_id}' AND (la.courseId = '{course_id}' OR ls.courseId = '{course_id}')
        AND la.activityType IN ('content_read', 'file_download', 'bookmark_add', 'note_create', 'help_request')
        """
        
        content_activities = self.db.select(
            "learning_activities la",
            "la.*, ls.title",
            where_clause
        )
        
        return content_activities
    
    def _get_notes_bookmarks(self, user_id, course_id):
        """Lấy ghi chú và bookmark của sinh viên"""
        
        # Lấy collaborative notes
        notes_where = f"""
        WHERE authorId = '{user_id}' AND courseId = '{course_id}'
        """
        
        notes = self.db.select(
            "collaborative_notes",
            "*",
            notes_where
        )
        
        return {'notes': notes}
    
    def _analyze_comprehensive_content_interaction(self, interaction_data, user_id, course_id):
        """
        Phân tích chi tiết tương tác với nội dung
        
        Args:
            interaction_data: Dict chứa tất cả dữ liệu tương tác
            user_id: ID sinh viên
            course_id: ID khóa học
            
        Returns:
            dict: Kết quả phân tích tương tác với nội dung
        """
        
        # 1. Phân tích tiến độ bài học
        lesson_analysis = self._analyze_lesson_progress(interaction_data['lesson_progress'])
        
        # 2. Phân tích tương tác video
        video_analysis = self._analyze_video_interactions(interaction_data['video_interactions'])
        
        # 3. Phân tích hoạt động nội dung
        content_analysis = self._analyze_content_activities(interaction_data['content_activities'])
        
        # 4. Phân tích ghi chú và bookmark
        notes_analysis = self._analyze_notes_bookmarks(interaction_data['notes_bookmarks'])
        
        # 5. Tính toán engagement score tổng thể
        engagement_score = self._calculate_engagement_score(
            lesson_analysis, video_analysis, content_analysis, notes_analysis
        )
        
        return {
            'lesson_progress_analysis': lesson_analysis,
            'video_interaction_analysis': video_analysis,
            'content_activity_analysis': content_analysis,
            'notes_bookmarks_analysis': notes_analysis,
            'overall_engagement_score': engagement_score
        }
    
    def _analyze_lesson_progress(self, lesson_progress):
        """Phân tích tiến độ bài học"""
        
        if not lesson_progress:
            return {
                'total_lessons': 0,
                'completion_stats': {},
                'time_analysis': {},
                'progress_patterns': {}
            }
        
        df = pd.DataFrame(lesson_progress)
        
        # Thống kê completion
        status_counts = df['status'].value_counts().to_dict()
        total_lessons = len(df)
        completed_lessons = status_counts.get('completed', 0)
        
        # Phân tích thời gian
        time_spent = df[df['timeSpent'].notna() & (df['timeSpent'] > 0)]
        avg_time_per_lesson = time_spent['timeSpent'].mean() if not time_spent.empty else 0
        total_time_spent = time_spent['timeSpent'].sum() if not time_spent.empty else 0
        
        # Phân tích theo loại bài học
        lesson_type_analysis = {}
        if 'lessonType' in df.columns:
            type_stats = df.groupby('lessonType').agg({
                'status': lambda x: (x == 'completed').sum(),
                'timeSpent': ['mean', 'sum'],
                'attempts': 'mean'
            }).round(2)
            
            for lesson_type in type_stats.index:
                lesson_type_analysis[lesson_type] = {
                    'completed': int(type_stats.loc[lesson_type, ('status', '<lambda>')]),
                    'avg_time': float(type_stats.loc[lesson_type, ('timeSpent', 'mean')]) if pd.notna(type_stats.loc[lesson_type, ('timeSpent', 'mean')]) else 0,
                    'total_time': float(type_stats.loc[lesson_type, ('timeSpent', 'sum')]) if pd.notna(type_stats.loc[lesson_type, ('timeSpent', 'sum')]) else 0,
                    'avg_attempts': float(type_stats.loc[lesson_type, ('attempts', 'mean')]) if pd.notna(type_stats.loc[lesson_type, ('attempts', 'mean')]) else 0
                }
        
        # Phân tích mô hình tiến độ
        progress_patterns = self._analyze_progress_patterns(df)
        
        return {
            'total_lessons': total_lessons,
            'completion_stats': {
                'completed': completed_lessons,
                'in_progress': status_counts.get('in_progress', 0),
                'not_started': status_counts.get('not_started', 0),
                'skipped': status_counts.get('skipped', 0),
                'completion_rate': round(completed_lessons / total_lessons * 100, 2) if total_lessons > 0 else 0
            },
            'time_analysis': {
                'total_time_spent': int(total_time_spent),
                'average_time_per_lesson': round(avg_time_per_lesson, 2),
                'time_efficiency': self._calculate_time_efficiency(df)
            },
            'lesson_type_analysis': lesson_type_analysis,
            'progress_patterns': progress_patterns
        }
    
    def _analyze_video_interactions(self, video_activities):
        """Phân tích tương tác video"""
        
        if not video_activities:
            return {
                'total_video_activities': 0,
                'engagement_patterns': {},
                'viewing_behavior': {}
            }
        
        df = pd.DataFrame(video_activities)
        
        # Thống kê các loại hoạt động video
        activity_counts = df['activityType'].value_counts().to_dict()
        
        # Phân tích hành vi xem video
        viewing_behavior = {}
        if 'video_play' in activity_counts:
            play_count = activity_counts['video_play']
            pause_count = activity_counts.get('video_pause', 0)
            seek_count = activity_counts.get('video_seek', 0)
            complete_count = activity_counts.get('video_complete', 0)
            
            # Tính các chỉ số
            pause_rate = pause_count / play_count if play_count > 0 else 0
            seek_rate = seek_count / play_count if play_count > 0 else 0
            completion_rate = complete_count / play_count if play_count > 0 else 0
            
            viewing_behavior = {
                'play_count': play_count,
                'pause_count': pause_count,
                'seek_count': seek_count,
                'complete_count': complete_count,
                'pause_rate': round(pause_rate, 3),
                'seek_rate': round(seek_rate, 3),
                'completion_rate': round(completion_rate, 3),
                'focus_score': round((1 - pause_rate - seek_rate) * 100, 2)
            }
        
        # Phân tích mô hình tương tác
        engagement_patterns = self._analyze_video_engagement_patterns(df)
        
        return {
            'total_video_activities': len(df),
            'activity_breakdown': activity_counts,
            'viewing_behavior': viewing_behavior,
            'engagement_patterns': engagement_patterns
        }
    
    def _analyze_content_activities(self, content_activities):
        """Phân tích các hoạt động tương tác với nội dung"""
        
        if not content_activities:
            return {
                'total_activities': 0,
                'activity_breakdown': {},
                'interaction_patterns': {}
            }
        
        df = pd.DataFrame(content_activities)
        
        # Thống kê các loại hoạt động
        activity_counts = df['activityType'].value_counts().to_dict()
        
        # Phân tích thời gian tương tác
        time_analysis = {}
        if 'duration' in df.columns:
            duration_data = df[df['duration'].notna() & (df['duration'] > 0)]
            if not duration_data.empty:
                time_analysis = {
                    'total_interaction_time': int(duration_data['duration'].sum()),
                    'average_interaction_time': round(duration_data['duration'].mean(), 2),
                    'median_interaction_time': round(duration_data['duration'].median(), 2)
                }
        
        # Phân tích mô hình tương tác
        interaction_patterns = self._analyze_interaction_patterns(df)
        
        return {
            'total_activities': len(df),
            'activity_breakdown': activity_counts,
            'time_analysis': time_analysis,
            'interaction_patterns': interaction_patterns
        }
    
    def _analyze_notes_bookmarks(self, notes_bookmarks):
        """Phân tích ghi chú và bookmark"""
        
        notes = notes_bookmarks.get('notes', [])
        
        if not notes:
            return {
                'total_notes': 0,
                'note_analysis': {},
                'content_engagement': {}
            }
        
        df = pd.DataFrame(notes)
        
        # Thống kê ghi chú
        note_stats = {
            'total_notes': len(df),
            'by_type': df['type'].value_counts().to_dict() if 'type' in df.columns else {},
            'by_status': df['status'].value_counts().to_dict() if 'status' in df.columns else {}
        }
        
        # Phân tích nội dung ghi chú
        content_analysis = {}
        if 'content' in df.columns:
            content_lengths = df['content'].str.len()
            content_analysis = {
                'average_length': round(content_lengths.mean(), 2) if not content_lengths.empty else 0,
                'total_characters': int(content_lengths.sum()) if not content_lengths.empty else 0,
                'longest_note': int(content_lengths.max()) if not content_lengths.empty else 0
            }
        
        return {
            'total_notes': len(df),
            'note_statistics': note_stats,
            'content_analysis': content_analysis,
            'engagement_level': self._calculate_note_engagement(df)
        }
    
    def _analyze_progress_patterns(self, df):
        """Phân tích mô hình tiến độ học tập"""
        
        # Sắp xếp theo orderIndex nếu có
        if 'orderIndex' in df.columns:
            df_sorted = df.sort_values('orderIndex')
            
            # Tìm các gap trong tiến độ
            completed_lessons = df_sorted[df_sorted['status'] == 'completed']
            if not completed_lessons.empty:
                completion_sequence = completed_lessons['orderIndex'].tolist()
                gaps = []
                for i in range(1, len(completion_sequence)):
                    if completion_sequence[i] - completion_sequence[i-1] > 1:
                        gaps.append(completion_sequence[i] - completion_sequence[i-1] - 1)
                
                return {
                    'sequential_completion': len(gaps) == 0,
                    'completion_gaps': gaps,
                    'average_gap': round(np.mean(gaps), 2) if gaps else 0
                }
        
        return {'pattern_analysis': 'insufficient_data'}
    
    def _calculate_time_efficiency(self, df):
        """Tính hiệu quả thời gian học"""
        
        if 'timeSpent' not in df.columns or 'estimatedDuration' not in df.columns:
            return {'efficiency_score': 0}
        
        # Lọc dữ liệu hợp lệ
        valid_data = df[
            (df['timeSpent'].notna()) & 
            (df['estimatedDuration'].notna()) & 
            (df['timeSpent'] > 0) & 
            (df['estimatedDuration'] > 0)
        ]
        
        if valid_data.empty:
            return {'efficiency_score': 0}
        
        # Tính tỷ lệ thời gian thực tế / thời gian dự kiến
        valid_data = valid_data.copy()
        valid_data['efficiency_ratio'] = valid_data['timeSpent'] / (valid_data['estimatedDuration'] * 60)
        
        avg_efficiency = valid_data['efficiency_ratio'].mean()
        efficiency_score = max(0, min(100, (2 - avg_efficiency) * 50))  
        
        return {
            'efficiency_score': round(efficiency_score, 2),
            'average_ratio': round(avg_efficiency, 2),
        }
    
    def _analyze_video_engagement_patterns(self, df):
        """Phân tích mô hình tương tác video"""
        
        if 'timestamp' not in df.columns:
            return {}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Phân tích theo giờ
        hourly_activity = df.groupby('hour').size().to_dict()
        
        # Tìm thời gian xem video cao điểm
        peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'hourly_distribution': hourly_activity,
            'peak_viewing_hours': [hour for hour, count in peak_hours],
            'most_active_hour': peak_hours[0][0] if peak_hours else None
        }
    
    def _analyze_interaction_patterns(self, df):
        """Phân tích mô hình tương tác với nội dung"""
        
        patterns = {}
        
        # Phân tích theo thời gian
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            
            patterns['temporal'] = {
                'hourly_distribution': df.groupby('hour').size().to_dict(),
                'daily_distribution': df.groupby('day_of_week').size().to_dict()
            }
        
        # Phân tích frequency
        if 'lessonId' in df.columns:
            lesson_interactions = df.groupby('lessonId').size()
            patterns['lesson_focus'] = {
                'most_interacted_lessons': lesson_interactions.nlargest(5).to_dict(),
                'average_interactions_per_lesson': round(lesson_interactions.mean(), 2)
            }
        
        return patterns
    
    def _calculate_note_engagement(self, df):
        """Tính mức độ tương tác qua ghi chú"""
        
        if df.empty:
            return {'engagement_level': 'low', 'score': 0}
        
    
        note_count = len(df)
        if 'content' in df.columns:
            avg_length = df['content'].str.len().mean()
        else:
            avg_length = 0
        
      
        count_score = min(note_count * 10, 50) 
        length_score = min(avg_length / 10, 50)  
        total_score = count_score + length_score
        
        if total_score >= 80:
            level = 'very_high'
        elif total_score >= 60:
            level = 'high'
        elif total_score >= 40:
            level = 'medium'
        elif total_score >= 20:
            level = 'low'
        else:
            level = 'very_low'
        
        return {
            'engagement_level': level,
            'score': round(total_score, 2),
            'note_count_contribution': round(count_score, 2),
            'content_length_contribution': round(length_score, 2)
        }
    
    def _calculate_engagement_score(self, lesson_analysis, video_analysis, content_analysis, notes_analysis):
        """Tính điểm engagement tổng thể"""
        
        scores = []
        weights = []
        
        # Lesson completion score (40%)
        lesson_score = lesson_analysis.get('completion_stats', {}).get('completion_rate', 0)
        scores.append(lesson_score)
        weights.append(0.4)
        
        # Video interaction score (25%)
        video_behavior = video_analysis.get('viewing_behavior', {})
        video_score = video_behavior.get('focus_score', 0)
        scores.append(video_score)
        weights.append(0.25)
        
        # Content interaction score (20%)
        content_score = min(content_analysis.get('total_activities', 0) * 5, 100)
        scores.append(content_score)
        weights.append(0.2)
        
        # Notes engagement score (15%)
        notes_score = notes_analysis.get('engagement_level_score', 0)
        if isinstance(notes_analysis.get('engagement_level', {}), dict):
            notes_score = notes_analysis.get('engagement_level', {}).get('score', 0)
        scores.append(notes_score)
        weights.append(0.15)
        
      
        total_score = sum(score * weight for score, weight in zip(scores, weights))
        
       
        if total_score >= 80:
            level = 'very_high'
        elif total_score >= 60:
            level = 'high'
        elif total_score >= 40:
            level = 'medium'
        elif total_score >= 20:
            level = 'low'
        else:
            level = 'very_low'
        
        return {
            'overall_score': round(total_score, 2),
            'engagement_level': level,
            'component_scores': {
                'lesson_completion': round(scores[0] * weights[0], 2),
                'video_interaction': round(scores[1] * weights[1], 2),
                'content_interaction': round(scores[2] * weights[2], 2),
                'notes_engagement': round(scores[3] * weights[3], 2)
            }
        }
        
class AITRACKING:
    def __init__(self, db):
        self.db = db
        self.data = AITrackingDataCollector(self.db)
    def extract_features_aitrack(self, data):
        """Chuẩn hóa dữ liệu từ AITracking thành feature vector"""
        features = []
        
        # Basic user info
        # features.append(1 if data.get('user_id') else 0)  # has_user_id
        # features.append(1 if data.get('course_id') else 0)  # has_course_id
        
        # Basic progress features
        basic_progress = (data.get('basic_progress') or [{}])[0]
        features.extend([
            basic_progress.get('totalLessons', 0),
            basic_progress.get('totalVideoDuration', 0),
            1 if basic_progress.get('level') == 'beginner' else 2 if basic_progress.get('level') == 'intermediate' else 3
        ])
        # print("tttttttttttttttttt",  basic_progress.get('totalLessons', 0))
        # Learning activities features
        learning_activities = data.get('learning_activities', {})
        activities = learning_activities.get('activity_summary', {})
        
        features.extend([
            learning_activities.get('total_activities', 0),
            # activities.get('note_create', {}).get('count', 0),
            activities.get('video_pause', {}).get('count', 0),
            activities.get('video_play', {}).get('count', 0),
            # activities.get('note_create', {}).get('avg_duration', 0),
            activities.get('video_pause', {}).get('avg_duration', 0),
            activities.get('video_play', {}).get('avg_duration', 0)
        ])
        
        # Engagement patterns
        engagement = learning_activities.get('engagement_patterns', {})
        streak = engagement.get('learning_streak', {})
        session = engagement.get('session_analysis', {})
        
        features.extend([
            streak.get('current_streak', 0),
            streak.get('longest_streak', 0),
            streak.get('total_learning_days', 0),
            session.get('total_sessions', 0),
            session.get('avg_activities_per_session', 0),
            session.get('avg_session_duration', 0),
            engagement.get('most_active_hour', 0)
        ])
        
        # Time analysis
        time_analysis = learning_activities.get('time_analysis', {})
        features.extend([
            time_analysis.get('total_learning_time', 0),
            time_analysis.get('average_session_duration', 0)
        ])
        
        # Learning behavior
        behavior = learning_activities.get('learning_behavior', {})
        focus = behavior.get('focus_patterns', {})
        video_eng = focus.get('video_engagement', {})
        completion = behavior.get('completion_patterns', {})
        interaction = behavior.get('content_interaction', {})
        
        features.extend([
            focus.get('focus_score', 0),
            video_eng.get('play_count', 0),
            video_eng.get('pause_count', 0),
            video_eng.get('completion_rate', 0),
            completion.get('completion_rate', 0),
            completion.get('total_completions', 0),
            interaction.get('interaction_count', 0)
        ])
        
        # Assessment performance
        assessment = data.get('assessment_performance', {})
        summary = assessment.get('assessment_summary', {})
        trends = assessment.get('performance_trends', {})
        difficulty = assessment.get('difficulty_analysis', {}).get('medium', {})
        time_perf = assessment.get('time_analysis', {})
        improvement = assessment.get('improvement_analysis', {})
        
        features.extend([
            summary.get('total_attempts', 0),
            round(float(summary.get('average_score', 0)),2),
            round(float(summary.get('average_percentage', 0)),2),
            summary.get('passed_attempts', 0),
            summary.get('failed_attempts', 0),
            float(summary.get('pass_rate', 0)),
            difficulty.get('total_questions', 0),
            difficulty.get('correct_answers', 0),
            float(difficulty.get('accuracy', 0)),
            time_perf.get('average_time', 0),
            float(improvement.get('average_improvement', 0)),
            float(improvement.get('consistency_score', 0)),
            float(improvement.get('total_improvement', 0))
        ])
        
        # Content interaction
        content = data.get('content_interaction', {})
        lesson_progress = content.get('lesson_progress_analysis', {})
        completion_stats = lesson_progress.get('completion_stats', {})
        time_stats = lesson_progress.get('time_analysis', {})
        video_interaction = content.get('video_interaction_analysis', {})
        video_behavior = video_interaction.get('viewing_behavior', {})
        overall = content.get('overall_engagement_score', {})
        
        features.extend([
            lesson_progress.get('total_lessons', 0),
            completion_stats.get('completed', 0),
            completion_stats.get('in_progress', 0),
            float(completion_stats.get('completion_rate', 0)),
            round((time_stats.get('total_time_spent', 0) / 3600), 2),
            round(time_stats.get('average_time_per_lesson', 0),2),
            video_interaction.get('total_video_activities', 0),
            video_behavior.get('play_count', 0),
            video_behavior.get('pause_count', 0),
            float(video_behavior.get('pause_rate', 0)),
            float(video_behavior.get('completion_rate', 0)),
            float(video_behavior.get('focus_score', 0)),
            float(overall.get('overall_score', 0))
        ])
        # print("Trước np.array", features)
        # print("Sau np.array", np.array(features))
        return np.array(features[:51])
    
    def train_performance_model(self):
        """Huấn luyện model dự đoán hiệu suất học tập"""
        
        where_clause = f""" 
        JOIN user_roles url ON url.user_id = ur.id
        JOIN roles rl ON rl.id = url.role_id
        JOIN enrollments en ON en.studentId = ur.id
        WHERE rl.name = 'student' AND en.courseId IS NOT NULL
        """
        students = self.db.select("users ur", " DISTINCT ur.id, en.courseId", where_clause)
        for student in students:
            studentId = student['id']
            courseId = student['courseId']
            data = self.data.collect_comprehensive_data(studentId,courseId)
            features = self.extract_features_aitrack(data)
            # print("Dữ liệu vào:", data)
            # print("Dữ liệu ra:", features)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     training_data, labels, test_size=0.2, random_state=42, stratify=labels
        # )
        
    
        # self.performance_scaler = StandardScaler()
        # X_train_scaled = self.performance_scaler.fit_transform(X_train)
        # X_test_scaled = self.performance_scaler.transform(X_test)
        
        
        # self.performance_model = RandomForestClassifier(
        #     n_estimators=100,
        #     max_depth=10,
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     class_weight='balanced',
        #     random_state=42,
        #     n_jobs=-1
        # )
        
        # self.performance_model.fit(X_train_scaled, y_train)
        
        
        # y_pred = self.performance_model.predict(X_test_scaled)
        # accuracy = accuracy_score(y_test, y_pred)
        # report = classification_report(y_test, y_pred)
        
        # print(f"Model Performance - Accuracy: {accuracy:.4f}")
        # print("Classification Report:")
        # print(report)
        
        # return {
        #     'accuracy': accuracy,
        #     'classification_report': report,
        #     'feature_importance': self.performance_model.feature_importances_
        # }
    
    def save_model(self, filename='aitrack_model.pkl'):
        """Lưu model đã huấn luyện"""
        import joblib
        
        model_data = {
            'performance_model': self.performance_model if hasattr(self, 'performance_model') else None,
            'performance_scaler': self.performance_scaler if hasattr(self, 'performance_scaler') else None
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Đã lưu model vào {filename}")
    
    def load_model(self, filename='aitrack_model.pkl'):
        """Tải model đã huấn luyện"""
        import joblib
        
        try:
            model_data = joblib.load(filename)
            self.performance_model = model_data.get('performance_model')
            self.performance_scaler = model_data.get('performance_scaler')
            print(f"✅ Đã tải model từ {filename}")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            return False
    
    def analyze_student(self, student_data):
        """Phân tích toàn diện một học sinh"""
        # Trích xuất features
        features = self.extract_features_aitrack(student_data)
        
        # Dự đoán hiệu suất
        performance_result = self.predict_performance(features)
        
        # Gợi ý chiến lược
        strategy_result = self.suggest_learning_strategy(features, performance_result)
        
        return {
            'student_id': student_data.get('user_id'),
            'course_id': student_data.get('course_id'),
            'features': features.tolist(),
            'performance_prediction': performance_result,
            'learning_strategy': strategy_result,
            'analysis_timestamp': student_data.get('collected_at')
        }
        
        
# class LearningUserProfile:
#     def __init__(self, data: Dict):
#         self.id_lesson = data.get('lesson_id')
#         # print(self.id_lesson)
#         self.total_questions = data.get('total_questions', 0)
#         self.correct_answers = data.get('correct_answers', 0)
#         self.overall_accuracy = data.get('overall_accuracy_decimal', 0.0)
#         self.confidence = data.get('confidence', 0.5)
#         self.strategy = data.get('strategy', 'INTENSIVE_FOUNDATION')
#         self.current_lesson = data.get('current_lesson', {})
#         self.lesson_history = data.get('lesson_history', [])
        
#     def example_data(self, db: DatabaseManager, user_id: str):
#         course_id = db.select("lessons", "courseId",f"WHERE id = '{self.id_lesson}'")  
#         where_clause = f"JOIN lessons ls ON lea.lessonId = ls.id WHERE lea.courseId = '{course_id[0].get('courseId')}'"
#         data_example = db.select("learning_activities lea","DISTINCT lea.lessonId, ls.orderIndex",where_clause )
#         # print(course_id)
        
#         print(data_example)
 
        
        
        
        
        
if __name__ == "__main__":
    db_manager = DatabaseManager()
    conn = db_manager.connect()

    if conn:
      
        analyzer = TestAnalyzer()
        analysis = analyzer.analyze_user_performance(db_manager, "user-student-12", "att-gu-12")
        Ai = LearningStrategyAI()
        cm = ContentRecommender()
        test = Ai.extract_features(analysis)
        # # print(f"🔍 test features: {test}")
        # # print("🔍 Calling predict_strategy...")
        result = Ai.predict_strategy(test)
        # # print(f"🔍 Raw result: {result}")
        # # print(f"🔍 Type of result: {type(result)}")

        strategy, confidence, sdf = result
        recomment = cm.recommend_lessons(db_manager, strategy, analysis)
        # # Test AITRACKING model
        # pretrack = AITrackingDataCollector(db_manager) 
        # student_data = pretrack.collect_comprehensive_data("user-student-01", "course-html-css")
        
        # # Khởi tạo AITRACKING model
        # aitrack_model = AITRACKING(db_manager)
        # t = aitrack_model.train_performance_model()
        # # Test extract features
        # features = aitrack_model.extract_features_aitrack(student_data)
        # print(f"✅ Features extracted: {len(features)} features")
        # print(f"Features: {features}")
        
        # # Test phân tích học sinh (không có model được train)
        # analysis_result = aitrack_model.analyze_student(student_data)
        # print(f"✅ Student Analysis:")
        # print(f"Student ID: {analysis_result['student_id']}")
        # print(f"Performance Level: {analysis_result['performance_prediction']['performance_level']}")
        # print(f"Strategy: {analysis_result['learning_strategy']['strategy']}")
        # print(f"Priority Areas: {analysis_result['learning_strategy']['priority_areas']}")
        
        
        # print(f"✅ Model trained with accuracy: {training_result['accuracy']:.4f}")
        
        
        # Lưu model
        # aitrack_model.save_model('demo_aitrack_model.pkl')
        # print("I'm Here ", testo)
        # learning = Learning_assessment(db_manager)
        # tesst = learning.learning_analytics_data("user-student-14", "lesson-html-tags")
        # rd = RandomForestLearninAttube()
        # t = rd.extract_features_lesson(tesst)
        # tr = rd.train()
        # r = rd.predict(tesst, return_proba = True)
        # print("\n=== Prediction Result ===")
        # print(f"Predicted attitude: {r['attitude']}")
        # print(f"Confidence: {r['confidence']:.2%}")
        # print("\nProbabilities for each class:")
        # for attitude, prob in r['probabilities'].items():
        #     print(f"  {attitude}: {prob:.2%}")
        
     
        
        # track = AITrackingDataCollector(db_manager)
        # test = track.collect_comprehensive_data("user-student-01","course-html-css")
        # print(test)
        # test = track.collect_comprehensive_data
        # cm = ContentRecommender(db_manager)
        # cm_def = cm.recommend_lessons()
        print("\n📊 KẾT QUẢ PHÂN TÍCH:")
        print("Has data:", analysis.get("has_data"))
        print("Total questions:", analysis.get("total_questions"))
        print("Total time:", analysis.get("total_time"), "s")
        print("Correct answers:", analysis.get("correct_answers"))
        print("Overall accuracy:", analysis.get("overall_accuracy_percent"), "%")
        print("Overall accuracy (decimal):", analysis.get("overall_accuracy"))
        print("Reason: ", result[2].get('reason') )

        print("\n📈 Hiệu suất theo danh mục:")
        print(json.dumps(analysis.get("assessment_attempt_performance", {}), indent=2, ensure_ascii=False))

        print("\n📉 Hiệu suất theo độ khó:")
        print(json.dumps(analysis.get("difficulty_performance", {}), indent=2, ensure_ascii=False))

        print("\n🚨 Danh mục yếu:")
        print(json.dumps(analysis.get("weak_categories", []), indent=2, ensure_ascii=False))

        print("\n📚 Cơ hội cải thiện:")
        print(json.dumps(analysis.get("avd_categories", []), indent=2, ensure_ascii=False))
        
           # ===== MODEL SAVE/LOAD DEMO =====
        # print("\n" + "="*60)
        # print("🔧 DEMO: Model Save/Load Operations")
        # print("="*60)
        
        # # Save models after training
        # print("\n1. Lưu models sau khi train...")
        ModelManager.save_all_models(Ai, "./models/")
        
        # # Show model info  
        # print("\n2. Thông tin models đã lưu:")
        # model_info = ModelManager.create_model_info("./models/")
        # print(json.dumps(model_info, indent=2, ensure_ascii=False, default=str))
        
        # # Load models from files
        # print("\n3. Load models từ files...")
        # loaded_strategy_ai, loaded_attitude_model = ModelManager.load_all_models("./models/")
        
        # # Test loaded models
        # print("\n4. Test models đã được load:")
        
        # # Test LearningStrategyAI
        # loaded_result = loaded_strategy_ai.predict_strategy(test)
        # print(f"   LearningStrategyAI: {loaded_result[0]} (confidence: {loaded_result[1]:.2%})")
        
        # # Test RandomForestLearninAttube  
        # loaded_attitude_result = loaded_attitude_model.predict(tesst, return_proba=True)
        # print(f"   RandomForestLearninAttube: {loaded_attitude_result['attitude']} (confidence: {loaded_attitude_result['confidence']:.2%})")
        
        # print("\n✅ Model save/load operations completed successfully!")
        # print("="*60)
        
        # print(f"🔍 strategy after assignment: {strategy}")
        # print(f"🔍 confidence after assignment: {confidence}")
        # # print(recomment)
        # # print(f"🔍 analysis after assignment: {sdf}")
        # # # print(strategy)
        # # print(recomment[0])
        if recomment:
            for idx, lesson in enumerate(recomment, 1):
                # print(lesson)
                # print(lesson.get('lesson_id'))
                print(f"\n{idx}. {lesson.get('title', 'N/A')}")
                print(f"   📖  Khóa học: {lesson.get('course_title', 'N/A')}")
                print(f"   🏷️  Danh mục: {lesson.get('category_name', 'N/A')}")
                print(f"   ⭐  Độ ưu tiên: {lesson.get('priority_score', 0):.2f}")
                print(f"   🤓  Độ khó: {lesson.get('difficulty')}")
                print(f"   💡  Lý do: {lesson.get('recommendation_reason', 'N/A')}")
                print(f"   💯  Tổng số câu hỏi của bài học: {lesson.get('total_question_lesson')}")
                print(f"   🧠  Độ chính xác trong bài học: {lesson.get('lesson_accuracy')}%")
                print(f"   🔗  Đường dẫn: {lesson.get('lesson_slug')}")
                # Thông tin bổ sung nếu có
                if 'error_count' in lesson:
                    print(f"   ❌  Số lỗi: {lesson['error_count']}")
                if 'questions_wrong' in lesson:
                    # print(f"   ❓  Câu sai: {', '.join(lesson['questions_wrong'])}")
                    print(f"   ❓  Câu sai: {' -> '.join(f'{q['title']}' for q in lesson['questions_wrong'])}")
                    
        #         learner_data = {
        #             'lesson_id': lesson.get('lesson_id'),
        #             'priority': lesson.get('priority_score'),
        #             'overall_accuracy_decimal': analysis.get("overall_accuracy"),
        #             'strategy': strategy,
        #             'current_lesson': {
        #                 'level': lesson.get('difficulty'),
        #                 'lesson_accuracy': lesson.get('lesson_accuracy')
        #             },
        #         }
                
                    

        # else:
        #     print("Không có bài học nào được đề xuất.")
        
        print("-" * 80)
        

        db_manager.close()