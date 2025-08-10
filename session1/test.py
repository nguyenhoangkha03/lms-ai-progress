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

                cursor.execute(sql)
                return cursor.fetchall()
        except Exception as e:
            print(f"❌ Lỗi SELECT: {e}")
            return []
            
    def select_raw(self, query):
        """Execute raw SQL query"""
        try:
            if not self.connection:
                print("❌ Chưa kết nối database!")
                return []
                
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            print(f"❌ Lỗi query: {e}")
            print(f"Query: {query[:100]}...")  # In 100 ký tự đầu của query để debug
            return []
            
    def close(self):
        """Đóng kết nối"""
        if self.connection:
            self.connection.close()
            print("🔐 Đã đóng kết nối database")

class AITrackingDataCollector:
    """
    Thu thập dữ liệu toàn diện cho AI Tracking dựa trên database schema chính xác
    """
    
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
            'learning_sessions': self._get_learning_sessions(user_id),
            
            # 4. ASSESSMENT PERFORMANCE - Kết quả kiểm tra
            'assessment_performance': self._get_assessment_performance(user_id, course_id),
            
            # 5. TIME PATTERNS - Mô hình thời gian học
            'time_patterns': self._get_time_patterns(user_id),
            
            # 6. ENGAGEMENT METRICS - Chỉ số tương tác
            'engagement_metrics': self._get_engagement_metrics(user_id, course_id),
            
            # 7. LEARNING STYLE - Phong cách học tập
            'learning_style': self._get_learning_style(user_id),
            
            # 8. CONTENT INTERACTION - Tương tác với nội dung
            'content_interaction': self._get_content_interaction(user_id, course_id),
            
            # 9. SOCIAL INTERACTION - Tương tác xã hội
            'social_interaction': self._get_social_interaction(user_id),
            
            # 10. HISTORICAL ANALYTICS - Dữ liệu phân tích lịch sử
            'historical_analytics': self._get_historical_analytics(user_id)
        }
        
        return data
    
    def _get_basic_progress(self, user_id: str, course_id: str = None):
        """Dữ liệu tiến độ cơ bản - CHÍNH XÁC theo database schema"""
        
        # Enrollments với đúng tên cột
        enrollments_query = f"""
            SELECT e.id, e.studentId, e.courseId, e.enrollmentDate, e.completionDate,
                   e.status, e.progressPercentage, e.lastAccessedAt, e.paymentStatus,
                   e.paymentAmount, e.totalTimeSpent, e.lessonsCompleted, e.totalLessons,
                   e.rating, e.review, e.reviewDate,
                   c.title as course_title, c.level, c.durationHours, c.price, c.isFree
            FROM enrollments e 
            JOIN courses c ON e.courseId = c.id
            WHERE e.studentId = '{user_id}'
            {f"AND e.courseId = '{course_id}'" if course_id else ""}
            ORDER BY e.enrollmentDate DESC
        """
        
        # Lesson Progress với đúng tên cột
        lesson_progress_query = f"""
            SELECT lp.id, lp.studentId, lp.lessonId, lp.enrollmentId,
                   lp.status, lp.completionDate, lp.timeSpent, lp.lastPosition,
                   lp.attempts, lp.score, lp.maxScore, lp.progressPercentage,
                   lp.firstAccessedAt, lp.lastAccessedAt, lp.isSkipped,
                   l.title as lesson_title, l.estimatedDuration, l.lessonType, l.orderIndex
            FROM lesson_progress lp
            JOIN lessons l ON lp.lessonId = l.id
            WHERE lp.studentId = '{user_id}'
            {f"AND EXISTS (SELECT 1 FROM enrollments e WHERE e.id = lp.enrollmentId AND e.courseId = '{course_id}')" if course_id else ""}
            ORDER BY lp.lastAccessedAt DESC
        """
        
        enrollments = self.db.select_raw(enrollments_query)
        lesson_progress = self.db.select_raw(lesson_progress_query)
        
        # Phân tích cơ bản
        analysis = self._analyze_basic_progress(enrollments, lesson_progress)
        
        return {
            'enrollments': enrollments,
            'lesson_progress': lesson_progress,
            'analysis': analysis
        }
    
    def _analyze_basic_progress(self, enrollments, lesson_progress):
        """Phân tích dữ liệu cơ bản"""
        if not enrollments:
            return {}
            
        enrollments_df = pd.DataFrame(enrollments)
        lesson_df = pd.DataFrame(lesson_progress) if lesson_progress else pd.DataFrame()
        
        # Thống kê enrollments
        total_enrollments = len(enrollments_df)
        completed_courses = len(enrollments_df[enrollments_df['status'] == 'completed'])
        dropped_courses = len(enrollments_df[enrollments_df['status'] == 'dropped'])
        avg_progress = enrollments_df['progressPercentage'].mean()
        
        # Thống kê lessons
        lesson_stats = {}
        if not lesson_df.empty:
            total_lessons = len(lesson_df)
            completed_lessons = len(lesson_df[lesson_df['status'] == 'completed'])
            skipped_lessons = len(lesson_df[lesson_df['isSkipped'] == 1])
            avg_lesson_progress = lesson_df['progressPercentage'].mean()
            
            lesson_stats = {
                'total_lessons_accessed': total_lessons,
                'completed_lessons': completed_lessons,
                'skipped_lessons': skipped_lessons,
                'completion_rate': (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0,
                'skip_rate': (skipped_lessons / total_lessons * 100) if total_lessons > 0 else 0,
                'avg_lesson_progress': avg_lesson_progress
            }
        
        # Warnings
        warnings = []
        if total_enrollments > 0:
            dropout_rate = (dropped_courses / total_enrollments * 100)
            if dropout_rate > 60:
                warnings.append("Tỷ lệ bỏ học cao (>60%)")
        
        if lesson_stats.get('skip_rate', 0) > 50:
            warnings.append("Tỷ lệ skip bài học cao (>50%)")
        
        if avg_progress < 30:
            warnings.append("Tiến độ học tập chậm (<30%)")
            
        return {
            'total_enrollments': total_enrollments,
            'completed_courses': completed_courses,
            'dropped_courses': dropped_courses,
            'dropout_rate': (dropped_courses / total_enrollments * 100) if total_enrollments > 0 else 0,
            'avg_course_progress': avg_progress,
            'lesson_statistics': lesson_stats,
            'warnings': warnings
        }
    
    def _get_learning_activities(self, user_id: str, course_id: str = None):
        """QUAN TRỌNG: Dữ liệu hành vi học tập chi tiết"""
        query = f"""
            SELECT la.id, la.studentId, la.courseId, la.lessonId, la.assessmentId,
                   la.activityType, la.sessionId, la.timestamp, la.duration,
                   la.metadata, la.ipAddress, la.userAgent, la.deviceType,
                   la.browser, la.operatingSystem, la.screenResolution,
                   la.timezone, la.location, la.trackingData
            FROM learning_activities la
            WHERE la.studentId = '{user_id}'
            {f"AND la.courseId = '{course_id}'" if course_id else ""}
            ORDER BY la.timestamp DESC
            LIMIT 1000
        """
        
        activities = self.db.select_raw(query)
        
        # Phân tích patterns từ activities
        activity_analysis = self._analyze_activity_patterns(activities)
        
        return {
            'raw_activities': activities,
            'patterns': activity_analysis
        }
    
    def _get_learning_sessions(self, user_id: str):
        """Dữ liệu phiên học tập"""
        query = f"""
            SELECT ls.id, ls.studentId, ls.sessionId, ls.startTime, ls.endTime,
                   ls.duration, ls.status, ls.pageViews, ls.activitiesCount,
                   ls.coursesAccessed, ls.lessonsAccessed, ls.assessmentsTaken,
                   ls.deviceType, ls.browser, ls.operatingSystem, ls.ipAddress,
                   ls.location, ls.engagementMetrics, ls.learningOutcomes,
                   ls.qualityIndicators, ls.metadata
            FROM learning_sessions ls
            WHERE ls.studentId = '{user_id}'
            ORDER BY ls.startTime DESC
            LIMIT 50
        """
        
        sessions = self.db.select_raw(query)
        
        return {
            'recent_sessions': sessions,
            'session_analysis': self._analyze_sessions(sessions)
        }
    
    def _get_assessment_performance(self, user_id: str, course_id: str = None):
        """Hiệu suất làm bài kiểm tra"""
        query = f"""
            SELECT aa.id, aa.studentId, aa.assessmentId, aa.attemptNumber,
                   aa.startedAt, aa.submittedAt, aa.gradingStatus,
                   aa.score, aa.maxScore, aa.percentage, aa.timeTaken,
                   aa.status, aa.answers, aa.feedback,
                   a.title as assessment_title, a.assessmentType, a.maxAttempts,
                   a.passingScore, a.totalPoints, a.timeLimit,
                   g.overallFeedback, g.gradedAt, g.isPublished
            FROM assessment_attempts aa
            JOIN assessments a ON aa.assessmentId = a.id
            LEFT JOIN grades g ON aa.id = g.attemptId
            WHERE aa.studentId = '{user_id}'
            {f"AND a.courseId = '{course_id}'" if course_id else ""}
            ORDER BY aa.submittedAt DESC
            LIMIT 100
        """
        
        attempts = self.db.select_raw(query)
        
        return {
            'attempts': attempts,
            'performance_trends': self._analyze_assessment_trends(attempts)
        }
    
    def _get_time_patterns(self, user_id: str):
        """Mô hình thời gian học tập"""
        query = f"""
            SELECT la.id, la.studentId, la.courseId, la.date,
                   la.totalTimeSpent, la.lessonsCompleted, la.assessmentsTaken,
                   la.averageScore, la.quizzesAttempted, la.quizzesPassed,
                   la.averageQuizScore, la.loginCount, la.videoWatchTime,
                   la.readingTime, la.discussionPosts, la.chatMessages,
                   la.mostActiveHour, la.engagementScore, la.progressPercentage,
                   la.performanceLevel, la.learningPattern, la.metadata
            FROM learning_analytics la
            WHERE la.studentId = '{user_id}'
            ORDER BY la.date DESC
            LIMIT 30
        """
        
        analytics = self.db.select_raw(query)
        
        return {
            'daily_analytics': analytics,
            'time_preferences': self._analyze_time_preferences(analytics)
        }
    
    def _get_engagement_metrics(self, user_id: str, course_id: str = None):
        """Chỉ số tương tác và engagement"""
        
        # Forum participation
        forum_query = f"""
            SELECT COUNT(*) as post_count, 
                   COALESCE(AVG(fp.upvoteCount), 0) as avg_upvotes,
                   COALESCE(AVG(fp.downvoteCount), 0) as avg_downvotes,
                   COALESCE(AVG(fp.score), 0) as avg_score
            FROM forum_posts fp
            WHERE fp.authorId = '{user_id}'
        """
        
        # Chat participation  
        chat_query = f"""
            SELECT COUNT(*) as message_count,
                   COUNT(DISTINCT cm.roomId) as rooms_participated
            FROM chat_messages cm
            LEFT JOIN chat_rooms cr ON cm.roomId = cr.id
            WHERE cm.senderId = '{user_id}'
            {f"AND cr.courseId = '{course_id}'" if course_id else ""}
        """
        
        # Video interactions từ learning_activities
        video_query = f"""
            SELECT COUNT(*) as video_activities,
                   COALESCE(AVG(la.duration), 0) as avg_interaction_time,
                   COUNT(CASE WHEN la.activityType = 'video_play' THEN 1 END) as play_count,
                   COUNT(CASE WHEN la.activityType = 'video_pause' THEN 1 END) as pause_count,
                   COUNT(CASE WHEN la.activityType = 'video_complete' THEN 1 END) as complete_count
            FROM learning_activities la
            WHERE la.studentId = '{user_id}' 
            AND la.activityType LIKE 'video_%'
            {f"AND la.courseId = '{course_id}'" if course_id else ""}
        """
        
        # File downloads
        download_query = f"""
            SELECT COUNT(*) as download_activities
            FROM learning_activities la
            WHERE la.studentId = '{user_id}'
            AND la.activityType = 'file_download'
            {f"AND la.courseId = '{course_id}'" if course_id else ""}
        """
        
        forum_data = self.db.select_raw(forum_query)[0] if self.db.select_raw(forum_query) else {}
        chat_data = self.db.select_raw(chat_query)[0] if self.db.select_raw(chat_query) else {}
        video_data = self.db.select_raw(video_query)[0] if self.db.select_raw(video_query) else {}
        download_data = self.db.select_raw(download_query)[0] if self.db.select_raw(download_query) else {}
        
        return {
            'forum_participation': forum_data,
            'chat_participation': chat_data,
            'video_engagement': video_data,
            'download_activity': download_data
        }
    
    def _get_learning_style(self, user_id: str):
        """Phong cách học tập"""
        query = f"""
            SELECT lsp.id, lsp.userId, lsp.primaryLearningStyle,
                   lsp.secondaryLearningStyle, lsp.preferredModality,
                   lsp.styleScores, lsp.learningPreferences,
                   lsp.cognitiveTraits, lsp.motivationalFactors,
                   lsp.confidenceLevel, lsp.interactionsAnalyzed,
                   lsp.lastAnalyzedAt, lsp.metadata
            FROM learning_style_profiles lsp
            WHERE lsp.userId = '{user_id}'
        """
        
        return self.db.select_raw(query)
    
    def _get_content_interaction(self, user_id: str, course_id: str = None):
        """Tương tác với nội dung"""
        
        # Notes created
        notes_query = f"""
            SELECT COUNT(*) as notes_count,
                   COUNT(CASE WHEN cn.type = 'shared' THEN 1 END) as shared_notes,
                   COUNT(CASE WHEN cn.type = 'collaborative' THEN 1 END) as collaborative_notes
            FROM collaborative_notes cn
            WHERE cn.authorId = '{user_id}'
            {f"AND cn.courseId = '{course_id}'" if course_id else ""}
        """
        
        # File interactions từ learning_activities
        file_query = f"""
            SELECT COUNT(CASE WHEN la.activityType = 'file_download' THEN 1 END) as downloads,
                   COUNT(CASE WHEN la.activityType = 'content_read' THEN 1 END) as content_reads,
                   COALESCE(AVG(la.duration), 0) as avg_content_time
            FROM learning_activities la
            WHERE la.studentId = '{user_id}'
            AND la.activityType IN ('file_download', 'content_read')
            {f"AND la.courseId = '{course_id}'" if course_id else ""}
        """
        
        # Bookmarks (từ lesson_progress.bookmarks)
        bookmark_query = f"""
            SELECT COUNT(*) as lessons_with_bookmarks
            FROM lesson_progress lp
            WHERE lp.studentId = '{user_id}'
            AND lp.bookmarks IS NOT NULL
            AND JSON_LENGTH(lp.bookmarks) > 0
        """
        
        notes_data = self.db.select_raw(notes_query)[0] if self.db.select_raw(notes_query) else {}
        file_data = self.db.select_raw(file_query)[0] if self.db.select_raw(file_query) else {}
        bookmark_data = self.db.select_raw(bookmark_query)[0] if self.db.select_raw(bookmark_query) else {}
        
        return {
            'notes_activity': notes_data,
            'file_interactions': file_data,
            'bookmarks': bookmark_data
        }
    
    def _get_social_interaction(self, user_id: str):
        """Tương tác xã hội"""
        
        # Study groups
        groups_query = f"""
            SELECT COUNT(*) as groups_joined,
                   COALESCE(AVG(sgm.contributionScore), 0) as avg_contribution,
                   COUNT(CASE WHEN sgm.role = 'owner' THEN 1 END) as groups_owned,
                   COUNT(CASE WHEN sgm.role = 'moderator' THEN 1 END) as groups_moderated
            FROM study_group_members sgm
            WHERE sgm.userId = '{user_id}' AND sgm.status = 'active'
        """
        
        # Peer reviews given
        reviews_given_query = f"""
            SELECT COUNT(*) as reviews_given,
                   COALESCE(AVG(prf.score), 0) as avg_score_given
            FROM peer_review_feedbacks prf
            WHERE prf.reviewerId = '{user_id}'
        """
        
        # Peer reviews received (submissions)
        reviews_received_query = f"""
            SELECT COUNT(*) as submissions_made,
                   COALESCE(AVG(prs.averageScore), 0) as avg_score_received,
                   COALESCE(AVG(prs.reviewsReceived), 0) as avg_reviews_per_submission
            FROM peer_review_submissions prs
            WHERE prs.submitterId = '{user_id}'
        """
        
        groups_data = self.db.select_raw(groups_query)[0] if self.db.select_raw(groups_query) else {}
        reviews_given = self.db.select_raw(reviews_given_query)[0] if self.db.select_raw(reviews_given_query) else {}
        reviews_received = self.db.select_raw(reviews_received_query)[0] if self.db.select_raw(reviews_received_query) else {}
        
        return {
            'study_groups': groups_data,
            'peer_reviews_given': reviews_given,
            'peer_reviews_received': reviews_received
        }
    
    def _get_historical_analytics(self, user_id: str):
        """Dữ liệu phân tích lịch sử"""
        
        # Dropout risk assessments
        risk_query = f"""
            SELECT dra.id, dra.studentId, dra.courseId, dra.assessmentDate,
                   dra.riskLevel, dra.riskProbability, dra.riskFactors,
                   dra.protectiveFactors, dra.interventionRequired,
                   dra.recommendedInterventions, dra.interventionPriority
            FROM dropout_risk_assessments dra
            WHERE dra.studentId = '{user_id}'
            ORDER BY dra.assessmentDate DESC
            LIMIT 10
        """
        
        # Performance predictions
        predictions_query = f"""
            SELECT pp.id, pp.studentId, pp.courseId, pp.predictionType,
                   pp.predictionDate, pp.targetDate, pp.predictedValue,
                   pp.confidenceScore, pp.riskLevel, pp.contributingFactors,
                   pp.actualValue, pp.accuracyScore, pp.isValidated
            FROM performance_predictions pp
            WHERE pp.studentId = '{user_id}'
            ORDER BY pp.predictionDate DESC
            LIMIT 10
        """
        
        # AI recommendations received
        recommendations_query = f"""
            SELECT ar.id, ar.studentId, ar.recommendationType, ar.contentId,
                   ar.title, ar.description, ar.reason, ar.confidenceScore,
                   ar.priority, ar.status, ar.createdAt, ar.interactedAt,
                   ar.interactionType, ar.userRating, ar.userFeedback
            FROM ai_recommendations ar
            WHERE ar.studentId = '{user_id}'
            ORDER BY ar.createdAt DESC
            LIMIT 20
        """
        
        risk_data = self.db.select_raw(risk_query)
        prediction_data = self.db.select_raw(predictions_query)
        recommendation_data = self.db.select_raw(recommendations_query)
        
        return {
            'risk_assessments': risk_data,
            'predictions': prediction_data,
            'ai_recommendations': recommendation_data
        }
    
    # Helper methods for analysis (giữ nguyên logic, chỉ sửa tên cột)
    def _analyze_activity_patterns(self, activities):
        """Phân tích mô hình từ learning activities"""
        if not activities:
            return {}
            
        df = pd.DataFrame(activities)
        
        patterns = {
            'activity_distribution': df['activityType'].value_counts().to_dict() if 'activityType' in df.columns else {},
            'device_usage': df['deviceType'].value_counts().to_dict() if 'deviceType' in df.columns else {},
            'browser_usage': df['browser'].value_counts().to_dict() if 'browser' in df.columns else {}
        }
        
        # Phân tích thời gian nếu có duration
        if 'duration' in df.columns and 'activityType' in df.columns:
            patterns['avg_duration_by_type'] = df.groupby('activityType')['duration'].mean().to_dict()
        
        # Phân tích theo giờ nếu có timestamp
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            patterns['hourly_patterns'] = df['hour'].value_counts().to_dict()
        
        return patterns
    
    def _analyze_sessions(self, sessions):
        """Phân tích sessions"""
        if not sessions:
            return {}
            
        df = pd.DataFrame(sessions)
        
        analysis = {
            'total_sessions': len(sessions),
            'avg_session_duration': df['duration'].mean() if 'duration' in df.columns else 0,
            'avg_activities_per_session': df['activitiesCount'].mean() if 'activitiesCount' in df.columns else 0,
            'avg_page_views': df['pageViews'].mean() if 'pageViews' in df.columns else 0
        }
        
        # Device analysis
        if 'deviceType' in df.columns:
            analysis['device_preferences'] = df['deviceType'].value_counts().to_dict()
            
        return analysis
    
    def _analyze_assessment_trends(self, attempts):
        """Phân tích xu hướng assessment"""
        if not attempts:
            return {}
            
        df = pd.DataFrame(attempts)
        
        analysis = {
            'total_attempts': len(attempts),
            'avg_score': df['score'].mean() if 'score' in df.columns and df['score'].notna().any() else 0,
            'avg_percentage': df['percentage'].mean() if 'percentage' in df.columns and df['percentage'].notna().any() else 0,
            'pass_rate': 0
        }
        
        # Calculate pass rate if data is available
        if 'percentage' in df.columns and 'passingScore' in df.columns and len(df) > 0:
            passed = df[df['percentage'] >= df['passingScore']]
            analysis['pass_rate'] = (len(passed) / len(df) * 100)
        
        # Trend analysis
        if 'submittedAt' in df.columns and 'percentage' in df.columns and len(df) > 1:
            df['submittedAt'] = pd.to_datetime(df['submittedAt'])
            df_sorted = df.sort_values('submittedAt')
            if len(df_sorted) >= 2:
                analysis['score_trend'] = self._calculate_trend(df_sorted['percentage'])
                
        return analysis
    
    def _analyze_time_preferences(self, analytics):
        """Phân tích preferences thời gian"""
        if not analytics:
            return {}
            
        df = pd.DataFrame(analytics)
        
        analysis = {
            'avg_daily_time': df['totalTimeSpent'].mean() if 'totalTimeSpent' in df.columns else 0,
            'avg_engagement_score': df['engagementScore'].mean() if 'engagementScore' in df.columns else 0,
            'dominant_performance_level': 'unknown'
        }
        
        # Get dominant performance level
        if 'performanceLevel' in df.columns and not df.empty:
            mode_result = df['performanceLevel'].mode()
            if not mode_result.empty:
                analysis['dominant_performance_level'] = mode_result[0]
        
        if 'mostActiveHour' in df.columns:
            analysis['preferred_hours'] = df['mostActiveHour'].value_counts().to_dict()
            
        if 'learningPattern' in df.columns:
            analysis['learning_patterns'] = df['learningPattern'].value_counts().to_dict()
            
        return analysis
    
    def _calculate_trend(self, series):
        """Tính xu hướng tăng/giảm"""
        if len(series) < 2:
            return 0
        return (series.iloc[-1] - series.iloc[0]) / len(series)

# Usage example với database chính xác
def track_comprehensive(user_id: str, course_id: str = None):
    """
    Sử dụng comprehensive tracking với database schema chính xác
    """
    # Khởi tạo database manager
    db_manager = DatabaseManager()
    
    # Kết nối database
    conn = db_manager.connect()
    
    if not conn:
        print("❌ Không thể kết nối database!")
        return None
    
    try:
        # Khởi tạo collector với database đã kết nối
        collector = AITrackingDataCollector(db_manager)
        
        # Thu thập dữ liệu
        comprehensive_data = collector.collect_comprehensive_data(user_id, course_id)
        
        # Lưu vào learning_analytics để cache nếu cần
        # save_to_learning_analytics(comprehensive_data)
        
        return comprehensive_data
        
    finally:
        # Đảm bảo đóng kết nối
        db_manager.close()

# Ví dụ sử dụng
if __name__ == "__main__":
    # Test với user từ database
    print("🚀 Bắt đầu thu thập dữ liệu AI Tracking...")
    user_data = track_comprehensive('user-student-01', 'course-html-css')
    
    if user_data:
        print("✅ Thu thập dữ liệu thành công!")
        print(json.dumps(user_data, indent=2, ensure_ascii=False))
    else:
        print("❌ Không thể thu thập dữ liệu!")