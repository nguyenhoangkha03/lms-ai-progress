import pymysql
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
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
    def analyze_user_performance(self, db: DatabaseManager, user_id: str, assessmentId : str):
        weak_categories = []
        avd_categories = []
        difficulty_stats = []
        question_ids = []
        lesson_recommendations = []  # Thêm để lưu gợi ý lesson
        difficulty_performance = {}
        
        where_clause = f"JOIN lessons ls ON qs.lessonId = ls.id WHERE qs.assessmentId = '{assessmentId}'"

        # Lấy danh sách question
        question_rows = db.select("questions qs", "qs.*, qs.id AS question_id, correctAnswer, difficulty, lessonId, ls.title AS lesson_title, ls.slug AS lesson_slug", where_clause)
        questions = {
            row['question_id']: {
                'question_title': row['questionText'],
                'lessonId': row['lessonId'],
                'point': int(row['points']),
                'orderIndex': int(row['orderIndex']),
                'lesson_title': row['lesson_title'],
                'lesson_slug': row['lesson_slug'],
                'correctAnswer': json.loads(row['correctAnswer']),
                'difficulty': row['difficulty']
            } for row in question_rows
        }
        
        # print("Here")
        
        # print(questions)
        lesson_process = db.select("lesson_progress lsp","lsp.*", f"WHERE studenId = '{user_id}'" )

        where_clause = f"""
        JOIN assessments a ON ast.assessmentId = a.id 
        LEFT JOIN lessons ls ON a.lessonId = ls.id
        LEFT JOIN courses cr ON cr.id = COALESCE(a.courseId, ls.courseId)
        LEFT JOIN categories cate ON cr.categoryId = cate.id 
        WHERE ast.studentId = '{user_id}' AND ast.assessmentId ='{assessmentId}'
        """

        assessment_attempts = db.select(
            "assessment_attempts ast",
            "ast.*, cate.name, cr.title AS course_title, ls.id AS lesson_id,COALESCE(a.courseId, ls.courseId), a.*",
            where_clause
        )
        # print(assessment_attempts)
        df = pd.DataFrame(assessment_attempts)

        # if df.empty:
        #     return {'has_data': False}

        # df['course_title'] = df.get('course_title', 'Unknown Course').fillna('Unknown Course')
        # df['lesson_title'] = df.get('lesson_title', 'Unknown Lesson').fillna('Unknown Lesson')

        assessment_attempt = df[["name","course_title"]]
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
            course_title = attempt.get('course_title')
            category_name = attempt.get('name')
            total_math = 0
            math = 0

            for qid, ans in student_answers.items():
                correct = questions.get(qid)
                question_math = correct.get('point')
                # print(question_math)
                if not correct:
                    continue
                    
                lesson_id = attempt.get('lessonId') or correct.get('lessonId')
                lesson_title = correct.get('lesson_title')
                lessonslug = correct.get('lesson_slug')
                # print(correct)
             
                
                # Simplified comparison logic
                # is_correct = (
                #     ans == correct['correctAnswer'] if isinstance(ans, list) and isinstance(correct['correctAnswer'], list)
                #     else (len(ans) == 1 and ans[0] in correct['correctAnswer']) if isinstance(ans, list)
                #     else ans in correct['correctAnswer'] if isinstance(correct['correctAnswer'], list)
                #     else ans == correct['correctAnswer']
                # )
                if isinstance(ans, list) and isinstance(correct['correctAnswer'], list):
                    is_correct = ans == correct['correctAnswer']
                    total_math = total_math + question_math
                elif isinstance(ans, list):
                    is_correct = len(ans) == 1 and ans[0] in correct['correctAnswer']
                    total_math = total_math + question_math
                elif isinstance(correct['correctAnswer'], list):
                    is_correct = ans in correct['correctAnswer']
                    total_math = total_math + question_math
                else:
                    is_correct = ans == correct['correctAnswer']
                    total_math = total_math + question_math
                if is_correct:
                    math = math + question_math
                    
                # print(total_math)

                
                
                
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
                question_ids.append(qid)
                difficulty_stats.append({
                    'id_question': qid,
                    'difficulty': correct['difficulty'],
                    'is_correct': int(is_correct)
                })
                
                # print(difficulty_stats)

        # Create lesson recommendations
        print(f"🔍 Total lesson_errors found: {len(lesson_errors)}")

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
            # print(row)
            cat_name = row["name"]
            course_name = row["course_title"]
            score_mean = (math/total_math) * 100
            # max_score = total_math
            # print(max_score)
            # print(score)
            # percentage = float(row['percentage'])
            
       
            
            # print(row_category)
            
            for difficulty, row_difficulty in difficulty_summary.iterrows():
                if total_accuracy < 100:
                    category_data = {
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
        
        # Feature names for interpretability
        self.feature_names = [
            'accuracy_norm', 'total_questions_norm', 'hint_usage_rate', 'total_time_norm',
            'easy_accuracy', 'medium_accuracy', 'hard_accuracy',
            'weak_categories_norm', 'min_weak_accuracy', 'avg_weak_accuracy',
            'easy_time_norm', 'medium_time_norm', 'hard_time_norm'
        ]
    
    def extract_features(self, performance_data: Dict) -> np.array:
        """Extract features từ performance analysis - FIXED VERSION"""
        
        if not performance_data.get('has_data'):
            return np.zeros(18)
        
        features = []
        
        # 1. Overall performance (4 features) - CHUẨN HÓA
        features.extend([
            performance_data.get('overall_accuracy'),
            # performance_data.get('overall_accuracy_percent', 0) / 100,  # [0,1]
            performance_data.get('total_questions', 0) / 100,          # [0,1]
            0.0,  # hint_usage_rate - không có trong data, dùng default
            performance_data.get('total_time', 60) / 120               # [0,1]
        ])
        
        # 2. Difficulty-based performance (3 features) - SỬA LỖI
        diff_perf = performance_data.get('difficulty_performance', {})
        
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in diff_perf and isinstance(diff_perf[difficulty], dict):
                # Lấy accuracy và chuẩn hóa về [0,1]
                accuracy = diff_perf[difficulty].get('accuracy', 50.0) / 100
                features.append(float(accuracy))
                features.append(1.0) #có làm
            else:
                features.append(0.5)  # Default
                features.append(0) #không làm
        
        # 3. Category weakness indicators (3 features) - CHUẨN HÓA
        weak_cats = performance_data.get('weak_categories', [])
        features.extend([
            len(weak_cats) / 5,  # Normalize số weak categories
            min([cat['accuracy_correct'] for cat in weak_cats]) / 100 if weak_cats else 1.0,
            np.mean([cat['accuracy_correct'] for cat in weak_cats]) / 100 if weak_cats else 1.0
        ])
        
        # 4. Time efficiency (3 features) - THÊM MỚI
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
        
        
        
        
    
        features = [float(f) if isinstance(f, (int, float)) else 0.5 for f in features]
        features = [max(0.0, min(f, 1.5)) for f in features]  
        
        print(f"📊 Final normalized features ({len(features)}): {[round(f, 3) for f in features]}")
        
        return np.array(features[:16])
    
    
    def train_model(self):
        X_train = []
        y_train = []
        
        np.random.seed(42)
        
        for i in range(200):
            if i < 40:  # INTENSIVE_FOUNDATION - Very weak
                # easy_acc = np.random.uniform(0.2, 0.6)
                # mediun_acc = np.random.uniform()
                # if easy_acc == 0.5:
                #     easy_flag = np.random.choice([0.0, 1.0])
                # else:
                #     easy_flag = 1.0
                features = [
                    # Overall 
                    np.random.uniform(0.2, 0.4),    # accuracy
                    np.random.uniform(0.02, 0.1),  # questions
                    np.random.uniform(0.6, 1.0),    # hints
                    np.random.uniform(1.2, 1.5),    # time
                    # Difficulty + flags 
                    np.random.uniform(0.4, 0.8),    # easy acc
                    1.0,       
                    np.random.uniform(0.0, 0.5),    # medium acc
                    np.random.choice([0.0, 1.0], p=[0.6, 0.4]),                            # medium flag
                    np.random.uniform(0.0, 0.2),    # hard acc
                    np.random.choice([0.0, 1.0], p=[0.7, 0.3]), # hard flag (maybe didn't try)
                    # Weakness (3)
                    np.random.uniform(0.6, 1.0),    # weak count
                    np.random.uniform(0.0, 0.2),    # min weak
                    np.random.uniform(0.1, 0.3),    # avg weak
                    # Time (3)
                    np.random.uniform(0.7, 1.0),    # easy_time: 70-100% (chậm với easy)
                    np.random.uniform(0.8, 1.0),    # medium_time: 80-100% (chậm với medium)
                    np.random.uniform(0.9, 1.0),    # hard_time: 90-100% (rất chậm với hard)
                ]
                X_train.append(features[:18])
                y_train.append(0)
                
            elif i < 80:  # GUIDED_PRACTICE
                medium_acc = np.random.uniform(0.2, 0.6)
                if medium_acc == 0.5:
                    medium_flag = np.random.choice([0.0, 1.0], p=[0.6, 0.4])
                else:
                    medium_flag = 1.0
                    
                hard_acc = np.random.uniform(0.0, 0.5)
                if hard_acc == 0.5:
                    hard_flag = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
                else:
                    hard_flag = 1.0
                features = [
                    # Overall (4)
                    np.random.uniform(0.4, 0.6),    # accuracy
                    np.random.uniform(0.1, 0.25),  # questions
                    np.random.uniform(0.2, 0.6),    # hints
                    np.random.uniform(0.5, 1.0),    # time
                    # Difficulty + flags (6)
                    np.random.uniform(0.6, 0.8),    # easy acc
                    1.0,       
                    medium_acc,    # medium acc
                    medium_flag,                            # medium flag
                    hard_acc,    # hard acc
                    hard_flag, # hard flag 
                    # Weakness (3)
                    np.random.uniform(0.4, 1.0),    # weak count
                    np.random.uniform(0.3, 0.6),    # min weak
                    np.random.uniform(0.2, 0.3),    # avg weak
                    # Time (3)
                    np.random.uniform(0.3, 0.6),    # easy_time: 70-100% (chậm với easy)
                    np.random.uniform(0.5, 0.8),    # medium_time: 80-100% (chậm với medium)
                    np.random.uniform(0.9, 1.0),    # hard_time: 90-100% (rất chậm với hard)
                ]
                X_train.append(features[:16])
                y_train.append(1)
                
            elif i < 120:  
                medium_acc = np.random.uniform(0.5, 0.7)
                if medium_acc == 0.5:
                    medium_flag = np.random.choice([0.0, 1.0], p=[0.3, 0.7])
                else:
                    medium_flag = 1.0
                    
                hard_acc = np.random.uniform(0.5, 0.8)
                if hard_acc == 0.5:
                    hard_flag = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
                else:
                    hard_flag = 1.0
                features = [
                    # Overall (4)
                    np.random.uniform(0.6, 0.9),    # accuracy
                    np.random.uniform(0.2, 0.35),  # questions
                    np.random.uniform(0.0, 0.6),    # hints
                    np.random.uniform(0.5, 1.0),    # time
                    # Difficulty + flags (6)
                    np.random.uniform(0.8, 1.0),    # easy acc
                    1.0,       
                    medium_acc,    # medium acc
                    medium_flag,   #medium flag
                    hard_acc,    # hard acc
                    hard_flag, # hard flag 
                    # Weakness (3)
                    np.random.uniform(0.2, 0.4),    # weak count
                    np.random.uniform(0.0, 0.4),    # min weak
                    np.random.uniform(0.3, 0.6),    # avg weak
                    # Time (3)
                    np.random.uniform(0.1, 0.5),    # easy_time: 70-100% (chậm với easy)
                    np.random.uniform(0.3, 0.8),    # medium_time: 80-100% (chậm với medium)
                    np.random.uniform(0.5, 1.0),    # hard_time: 90-100% (rất chậm với hard)
                ]
                X_train.append(features[:16])
                y_train.append(2)
            elif i < 160:
                medium_acc = np.random.uniform(0.6, 1.0)
                if medium_acc == 0.5:
                    medium_flag = np.random.choice([0.0, 1.0], p=[0.2, 0.8])
                else:
                    medium_flag = 1.0
                    
                hard_acc = np.random.uniform(0.6, 1.0)
                if hard_acc == 0.5:
                    hard_flag = np.random.choice([0.0, 1.0], p=[0.6, 0.4])
                else:
                    hard_flag = 1.0
                features = [
                    # Overall (4)
                    np.random.uniform(0.8, 1.0),    # accuracy
                    np.random.uniform(0.3, 0.4),  # questions
                    np.random.uniform(0.0, 0.3),    # hints
                    np.random.uniform(0.3, 0.5),    # time
                    # Difficulty + flags (6)
                    np.random.uniform(0.8, 1.0),    # easy acc
                    1.0,       
                    medium_acc,    # medium acc
                    medium_flag,   #medium flag
                    hard_acc,    # hard acc
                    hard_flag, # hard flag 
                    # Weakness (3)
                    np.random.uniform(0.0, 0.2),    # weak count
                    np.random.uniform(0.0, 0.2),    # min weak
                    np.random.uniform(0.0, 0.6),    # avg weak
                    # Time (3)
                    np.random.uniform(0.1, 0.5),    # easy_time: 70-100% (chậm với easy)
                    np.random.uniform(0.1, 0.5),    # medium_time: 80-100% (chậm với medium)
                    np.random.uniform(0.2, 0.6),    # hard_time: 90-100% (rất chậm với hard)
                ]
                X_train.append(features[:16])
                y_train.append(3)
            else:
                medium_acc = np.random.uniform(0.5, 0.7)
                if medium_acc == 0.5:
                    medium_flag = np.random.choice([0.0, 1.0], p=[0.3, 0.7])
                else:
                    medium_flag = 1.0
                    
                hard_acc = np.random.uniform(0.5, 1.0)
                if hard_acc == 0.5:
                    hard_flag = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
                else:
                    hard_flag = 1.0
                features = [
                    # Overall (4)
                    np.random.uniform(0.6, 0.9),    # accuracy
                    np.random.uniform(0.2, 0.35),  # questions
                    np.random.uniform(0.0, 0.6),    # hints
                    np.random.uniform(0.5, 1.0),    # time
                    # Difficulty + flags (6)
                    np.random.uniform(0.8, 1.0),    # easy acc
                    1.0,       
                    medium_acc,    # medium acc
                    medium_flag,   #medium flag
                    hard_acc,    # hard acc
                    hard_flag, # hard flag 
                    # Weakness (3)
                    np.random.uniform(0.2, 0.4),    # weak count
                    np.random.uniform(0.0, 0.4),    # min weak
                    np.random.uniform(0.3, 0.6),    # avg weak
                    # Time (3)
                    np.random.uniform(0.1, 0.5),    
                    np.random.uniform(0.3, 0.8),   
                    np.random.uniform(0.5, 1.0),    
                ]
                X_train.append(features[:16])
                y_train.append(4)
                
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
                
    def predict_strategy(self, features: np.array) -> Tuple[str, float, Dict]:
        """Predict learning strategy with rules-based override"""
        if not self.is_trained:
            self.train_model()
        
        # Extract key metrics
        accuracy = features[0]
        # print(features.reshape(1, -1))
        
        
        # hint_usage = features[0]
        
        # Rules-based override for clear cases
        if accuracy < 0.4:
            return 'INTENSIVE_FOUNDATION', 0.95, {'reason': 'Very low accuracy'}
        elif accuracy > 0.5:
            return 'CHALLENGE_MODE', 0.90, {'reason': 'High performance'}
        elif accuracy > 0.7 and accuracy <= 0.85:
            return 'ADAPTIVE_LEARNING', 0.85, {'reason': 'Good performance'}
        
        # Otherwise use ML model
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        # print("here", features_scaled)
        strategy_id = self.model.predict(features_scaled)[0]
        # print(strategy_id)
        probabilities = self.model.predict_proba(features_scaled)[0]
        # print(probabilities)
        
        strategy = self.STRATEGIES[strategy_id]
        confidence = probabilities[strategy_id]
        
        return strategy, confidence, {
            'all_probabilities': {
                self.STRATEGIES[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
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
        available_courses = db_manager.select("courses", "DISTINCT level")
        available_levels = [course['level'] for course in available_courses]
        # print(strategy)
        # print(available_levels)
        # print(strategy_criteria['levels'])
        
        # Filter desired levels based on availability
        actual_levels = ([level for level in strategy_criteria['levels'] if level in available_levels], '')
        
        # print("Here")
        # print(actual_levels)
        # Extract categories from user_performance
        weak_categories = user_performance.get('weak_categories', [])
        # print("Herreeeeeeee")
        # print(weak_categories)
        avd_categories = user_performance.get('avd_categories', [])
        recomment = user_performance.get('lesson_recommendations', [])
        
        if weak_categories:
            for idx, data in enumerate(weak_categories[:5]):
                for lesson_data in recomment:
                    print(recomment)
                    if lesson_data.get('lesson_accuracy') < 90:
                        recomment_dict = {
                                'lesson_id': lesson_data.get('lesson_id'),
                                'title': f"Ôn tập {lesson_data.get('lesson_title')}",
                                'course_title': data.get('course_name'),
                                'level': actual_levels,
                                'category_name': data['category'],
                                'accuracy': data.get('accuracy_correct', 0),
                                'error_count': lesson_data.get('error_count', 0),
                                'difficulty': data.get('difficulty_question', 'medium'),
                                'priority': data.get('priority', 'MEDIUM'),
                                'correct_answers': data.get('correct_answers', 0), 
                                'total_question_lesson': lesson_data.get('total_question_lesson', 0),
                                'lesson_slug': lesson_data.get('lesson_slug', 'N/A'),
                                'lesson_accuracy': lesson_data.get('lesson_accuracy', 0),
                                'difficulties_affected': lesson_data.get('difficulties_affected', []),
                                'questions_wrong': lesson_data.get('questions_wrong', []),
                                'recommendation_reason': lesson_data.get('reason', ''),
                            }
                            
                            # Calculate priority score based on strategy and existing data
                        priority_score = self._calculate_priority_from_performance(
                            recomment_dict,
                            strategy_criteria,
                            user_performance
                        )
                        
                        recomment_dict['priority_score'] = priority_score
                        recommendations.append(recomment_dict)
                        
                        
                        # print(recommendations)
                    
        elif avd_categories:
            for idx, data in enumerate(avd_categories[:5]):
                for lesson_data in recomment:
                    if lesson_data.get('lesson_accuracy') < 90:
                        recomment_dict = {
                            'lesson_id': lesson_data.get('lesson_id'),
                            'title': f"Ôn tập {lesson_data.get('lesson_title')}",
                            'course_title': data.get('course_name'),
                            'level': actual_levels,
                            'category_name': data['category'],
                            'accuracy': data.get('accuracy_correct', 0),
                            'error_count': lesson_data.get('error_count', 0),
                            'difficulty': data.get('difficulty_question', 'medium'),
                            'priority': data.get('priority', 'MEDIUM'),
                            'correct_answers': data.get('correct_answers', 0),
                            'lesson_slug': lesson_data.get('lesson_slug', 'N/A'),
                            'lesson_accuracy': lesson_data.get('lesson_accuracy', 0),
                            'difficulties_affected': lesson_data.get('difficulties_affected', []),
                            'questions_wrong': lesson_data.get('questions_wrong', []),
                            'recommendation_reason': lesson_data.get('reason', ''),
                        }
                        
                            # Calculate priority score based on strategy and existing data
                        priority_score = self._calculate_priority_from_performance(
                            recomment_dict,
                            strategy_criteria,
                            user_performance
                        )
                        
                        recomment_dict['priority_score'] = priority_score
                        recommendations.append(recomment_dict)
                        
                    
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
            # print(recommendations)
            
                
                
                # Limit recommendations
        return recommendations[:5]
        
        # # Determine target courses based on categories
        # target_courses = []
        # if weak_categories:
        #     target_courses = [cat['course_name'] if isinstance(cat, dict) else cat for cat in weak_categories[:3]]
        # elif avd_categories:  
        #     target_courses = [cat['course_name'] if isinstance(cat, dict) else cat for cat in avd_categories[:3]]
        
        # # Build query conditions
        # if target_courses and actual_levels:
        #     course_conditions = " OR ".join([f"cr.title = '{course}'" for course in target_courses])
        #     level_conditions = " OR ".join([f"cr.level = '{level}'" for level in actual_levels])
            
        #     where_clause = f"""
        #     JOIN courses cr ON ls.courseId = cr.id
        #     WHERE ({course_conditions}) AND ({level_conditions})
        #     ORDER BY ls.id
        #     """
            
        #     lessons = db_manager.select("lessons ls", "ls.title AS lesson_title", where_clause)
            
        #     # Process lessons into recommendations
        #     for lesson in lessons:
        #         recommendations.append({
        #             'lesson_title': lesson['lesson_title'],
        #             'strategy': strategy,
        #             'focus': strategy_criteria['focus'],
        #             'order': strategy_criteria['order']
        #         })
        
        
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
        analysis = analyzer.analyze_user_performance(db_manager, "user-student-01", "assess-html-1")
        Ai = LearningStrategyAI()
        cm = ContentRecommender()
        test = Ai.extract_features(analysis)
        # print(f"🔍 test features: {test}")
        # print("🔍 Calling predict_strategy...")
        result = Ai.predict_strategy(test)
        # print(f"🔍 Raw result: {result}")
        # print(f"🔍 Type of result: {type(result)}")

        strategy, confidence, sdf = result
        recomment = cm.recommend_lessons(db_manager, strategy, analysis)
        # # cm = ContentRecommender(db_manager)
        # # cm_def = cm.recommend_lessons()
        print("\n📊 KẾT QUẢ PHÂN TÍCH:")
        print("Has data:", analysis.get("has_data"))
        print("Total questions:", analysis.get("total_questions"))
        print("Total time:", analysis.get("total_time"), "s")
        print("Correct answers:", analysis.get("correct_answers"))
        print("Overall accuracy:", analysis.get("overall_accuracy_percent"), "%")
        print("Overall accuracy (decimal):", analysis.get("overall_accuracy"))
        print("Reason: ", result[2].get('reason') )

        # print("\n📈 Hiệu suất theo danh mục:")
        # print(json.dumps(analysis.get("assessment_attempt_performance", {}), indent=2, ensure_ascii=False))

        # print("\n📉 Hiệu suất theo độ khó:")
        # print(json.dumps(analysis.get("difficulty_performance", {}), indent=2, ensure_ascii=False))

        # print("\n🚨 Danh mục yếu:")
        # print(json.dumps(analysis.get("weak_categories", []), indent=2, ensure_ascii=False))

        # print("\n📚 Cơ hội cải thiện:")
        # print(json.dumps(analysis.get("avd_categories", []), indent=2, ensure_ascii=False))
        
        print(f"🔍 strategy after assignment: {strategy}")
        print(f"🔍 confidence after assignment: {confidence}")
        # print(recomment)
        # print(f"🔍 analysis after assignment: {sdf}")
        # # print(strategy)
        # print(recomment[0])
        if recomment:
            for idx, lesson in enumerate(recomment, 1):
                # print(lesson)
                print(lesson.get('lesson_id'))
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
                    
                learner_data = {
                    'lesson_id': lesson.get('lesson_id'),
                    'priority': lesson.get('priority_score'),
                    'overall_accuracy_decimal': analysis.get("overall_accuracy"),
                    'strategy': strategy,
                    'current_lesson': {
                        'level': lesson.get('difficulty'),
                        'lesson_accuracy': lesson.get('lesson_accuracy')
                    },
                }
                
                NN = LearningUserProfile(learner_data)
                test = NN.example_data(db_manager, "user-student-01")
                    

        else:
            print("Không có bài học nào được đề xuất.")
        
        print("-" * 80)
        

        db_manager.close()