import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import json

# Experience tuple cho replay buffer
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class LearnerProfile:
    """Lớp đại diện cho hồ sơ người học"""
    def __init__(self, data: Dict):
        self.total_questions = data.get('total_questions', 0)
        self.correct_answers = data.get('correct_answers', 0)
        self.overall_accuracy = data.get('overall_accuracy_decimal', 0.0)
        self.confidence = data.get('confidence', 0.5)
        self.strategy = data.get('strategy', 'INTENSIVE_FOUNDATION')
        self.current_lesson = data.get('current_lesson', {})
        self.lesson_history = data.get('lesson_history', [])
        
    def to_state_vector(self) -> np.ndarray:
        """Chuyển đổi hồ sơ thành vector trạng thái cho neural network"""
        # Mã hóa strategy thành one-hot
        strategies = ['INTENSIVE_FOUNDATION', 'BALANCED', 'ADVANCED', 'REVIEW']
        strategy_encoding = [1 if self.strategy == s else 0 for s in strategies]
        
        # Mã hóa level hiện tại
        levels = ['beginner', 'intermediate', 'advanced']
        current_level = self.current_lesson.get('level', ['beginner'])[0]
        level_encoding = [1 if current_level == l else 0 for l in levels]
        
        # Mã hóa category
        categories = ['Front-End', 'Back-End', 'Database', 'DevOps', 'Mobile']
        current_category = self.current_lesson.get('category_name', 'Front-End')
        category_encoding = [1 if current_category == c else 0 for c in categories]
        
        # Vector đặc trưng
        state = np.array([
            self.overall_accuracy,
            self.confidence,
            self.total_questions / 100.0,  # Chuẩn hóa
            self.correct_answers / max(self.total_questions, 1),
            self.current_lesson.get('lesson_accuracy', 0.0),
            self.current_lesson.get('error_count', 0) / 10.0,  # Chuẩn hóa
            len(self.lesson_history) / 50.0,  # Chuẩn hóa số bài đã học
            *strategy_encoding,
            *level_encoding,
            *category_encoding
        ])
        
        return state

class DQN(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 128, 64]):
        super(DQN, self).__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class LessonRecommenderDQN:
    """Hệ thống đề xuất bài học sử dụng Deep Q-Learning"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Danh sách bài học có sẵn
        self.available_lessons = []
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights từ Q-network sang target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Lưu experience vào memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Chọn action dựa trên epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  # Tắt gradient để tăng hiệu suất
            q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().numpy())
    
    def calculate_reward(self, learner: LearnerProfile, 
                        selected_lesson: Dict, 
                        performance_after: Optional[Dict] = None) -> float:
        """Tính reward dựa trên hiệu quả học tập"""
        reward = 0.0
        
        # Reward cơ bản dựa trên độ phù hợp của level
        current_accuracy = learner.overall_accuracy
        lesson_level = selected_lesson.get('level', ['beginner'])[0]
        
        if current_accuracy < 0.5 and lesson_level == 'beginner':
            reward += 1.0
        elif 0.5 <= current_accuracy < 0.8 and lesson_level == 'intermediate':
            reward += 1.0
        elif current_accuracy >= 0.8 and lesson_level == 'advanced':
            reward += 1.0
        else:
            reward -= 0.5
            
        # Reward dựa trên chiến lược học tập
        if learner.strategy == 'INTENSIVE_FOUNDATION':
            if selected_lesson.get('priority') == 'high':
                reward += 0.5
                
        # Nếu có dữ liệu performance sau khi học
        if performance_after:
            accuracy_improvement = performance_after.get('accuracy', 0) - current_accuracy
            reward += accuracy_improvement * 2.0
            
        return reward
    
    def replay(self, batch_size: int = 32):
        """Train network với experiences từ memory"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        # Chuyển đổi thành numpy arrays trước khi tạo tensors
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # Tạo tensors từ numpy arrays
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Forward pass
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Đảm bảo cùng shape cho MSE loss
        loss = nn.MSELoss()(current_q_values.squeeze(1), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def recommend_next_lesson(self, learner_profile: LearnerProfile, 
                            available_lessons: List[Dict]) -> Dict:
        """Đề xuất bài học tiếp theo cho người học"""
        self.available_lessons = available_lessons
        
        # Chuyển đổi profile thành state vector
        state = learner_profile.to_state_vector()
        
        # Chọn action (index của bài học)
        action = self.act(state, training=False)
        
        # Đảm bảo action hợp lệ
        if action >= len(available_lessons):
            action = action % len(available_lessons)
            
        recommended_lesson = available_lessons[action]
        
        # Thêm lý do đề xuất
        recommendation_info = {
            'lesson': recommended_lesson,
            'reason': self._generate_recommendation_reason(learner_profile, recommended_lesson),
            'confidence_score': self._calculate_confidence_score(state, action)
        }
        
        return recommendation_info
    
    def _generate_recommendation_reason(self, profile: LearnerProfile, lesson: Dict) -> str:
        """Tạo lý do đề xuất bài học"""
        reasons = []
        
        if profile.overall_accuracy < 0.5:
            reasons.append("Độ chính xác tổng thể còn thấp, cần củng cố kiến thức cơ bản")
            
        if profile.strategy == 'INTENSIVE_FOUNDATION':
            reasons.append("Chiến lược học tập tập trung vào nền tảng")
            
        if lesson.get('priority') == 'high':
            reasons.append("Bài học có độ ưu tiên cao cho trình độ hiện tại")
            
        return ". ".join(reasons) if reasons else "Bài học phù hợp với tiến độ học tập hiện tại"
    
    def _calculate_confidence_score(self, state: np.ndarray, action: int) -> float:
        """Tính điểm tin cậy của đề xuất"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Softmax để chuyển thành xác suất
        exp_values = np.exp(q_values - np.max(q_values))
        probabilities = exp_values / exp_values.sum()
        
        return float(probabilities[action])
    
    def save_model(self, filepath: str):
        """Lưu model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# Ví dụ sử dụng
def main():
    # Dữ liệu người học mẫu
    learner_data = {
        'total_questions': 3,
        'correct_answers': 1,
        'overall_accuracy_decimal': 0.3333,
        'confidence': 0.95,
        'strategy': 'INTENSIVE_FOUNDATION',
        'current_lesson': {
            'title': 'Ôn tập Các thẻ tiêu đề và đoạn văn',
            'course_title': 'HTML & CSS cho người mới bắt đầu',
            'level': ['beginner'],
            'category_name': 'Front-End',
            'accuracy': 33.0,
            'error_count': 2,
            'lesson_accuracy': 0.0
        },
        'lesson_history': []
    }
    
    # Danh sách bài học có sẵn (ví dụ)
    available_lessons = [
        {
            'title': 'Thẻ HTML cơ bản',
            'level': ['beginner'],
            'category_name': 'Front-End',
            'priority': 'high',
            'difficulty': 'easy'
        },
        {
            'title': 'CSS Selectors',
            'level': ['beginner'],
            'category_name': 'Front-End',
            'priority': 'medium',
            'difficulty': 'medium'
        },
        {
            'title': 'Box Model trong CSS',
            'level': ['intermediate'],
            'category_name': 'Front-End',
            'priority': 'low',
            'difficulty': 'medium'
        }
    ]
    
    # Khởi tạo hệ thống
    state_size = 19  # Kích thước vector trạng thái
    action_size = len(available_lessons)
    
    recommender = LessonRecommenderDQN(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.95
    )
    
    # Tạo profile người học
    learner = LearnerProfile(learner_data)
    
    # Đề xuất bài học tiếp theo
    recommendation = recommender.recommend_next_lesson(learner, available_lessons)
    
    print("=== ĐỀ XUẤT BÀI HỌC TIẾP THEO ===")
    print(f"Bài học: {recommendation['lesson']['title']}")
    print(f"Level: {recommendation['lesson']['level'][0]}")
    print(f"Lý do: {recommendation['reason']}")
    print(f"Độ tin cậy: {recommendation['confidence_score']:.2%}")
    
    # Training example (giả lập)
    print("\n=== TRAINING EXAMPLE ===")
    
    # Giả lập học bài và nhận feedback
    state = learner.to_state_vector()
    action = 0  # Chọn bài học đầu tiên
    
    # Giả sử sau khi học, accuracy tăng lên
    performance_after = {'accuracy': 0.5}
    reward = recommender.calculate_reward(learner, available_lessons[action], performance_after)
    
    # Cập nhật state mới
    learner.overall_accuracy = 0.5
    next_state = learner.to_state_vector()
    
    # Lưu experience
    recommender.remember(state, action, reward, next_state, False)
    
    # Train
    recommender.replay(batch_size=1)
    
    print(f"Reward nhận được: {reward}")
    print(f"Epsilon hiện tại: {recommender.epsilon:.4f}")

if __name__ == "__main__":
    main()