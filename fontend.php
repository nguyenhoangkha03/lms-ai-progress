<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid AI Tracking Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gauge {
            position: relative;
            display: inline-block;
        }
        
        .comparison-bar {
            position: relative;
            background: #e5e7eb;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .comparison-bar .personal {
            position: absolute;
            height: 100%;
            background: #3b82f6;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .comparison-bar .benchmark {
            position: absolute;
            width: 2px;
            height: 16px;
            background: #ef4444;
            top: -4px;
            transform: translateX(-50%);
        }
    </style>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto p-6">
        <!-- Header -->
        <div class="bg-white rounded-lg shadow-sm p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">Hybrid AI Tracking Dashboard</h1>
            <p class="text-gray-600">Kết hợp phân tích Tổng thể + Cá nhân hóa</p>
            <div class="flex gap-4 mt-4">
                <button onclick="switchView('personal')" id="btn-personal" class="px-4 py-2 bg-blue-500 text-white rounded-lg">
                    👤 Cá nhân
                </button>
                <button onclick="switchView('cohort')" id="btn-cohort" class="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg">
                    👥 Nhóm tương tự
                </button>
                <button onclick="switchView('global')" id="btn-global" class="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg">
                    🌍 Toàn hệ thống
                </button>
            </div>
        </div>

        <!-- Student Info -->
        <div class="bg-white rounded-lg shadow-sm p-6 mb-6">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                    <h3 class="text-sm font-semibold text-gray-500">Học viên</h3>
                    <p class="text-xl font-bold">Nguyễn Văn A</p>
                    <p class="text-sm text-gray-600">ID: user-student-01</p>
                </div>
                <div>
                    <h3 class="text-sm font-semibold text-gray-500">Cohort</h3>
                    <p class="text-xl font-bold">Normal Pace - Moderately Engaged</p>
                    <p class="text-sm text-gray-600">150 học viên tương tự</p>
                </div>
                <div>
                    <h3 class="text-sm font-semibold text-gray-500">Risk Level</h3>
                    <div class="flex items-center gap-2">
                        <span class="text-xl font-bold text-orange-500">Medium</span>
                        <span class="text-sm text-gray-600">(45/100)</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Comparison Metrics -->
            <div class="bg-white rounded-lg shadow-sm p-6">
                <h2 class="text-xl font-bold mb-4">📊 So sánh Metrics</h2>
                <div class="space-y-6">
                    <!-- Progress -->
                    <div>
                        <div class="flex justify-between mb-2">
                            <span class="font-medium">Tiến độ học tập</span>
                            <span class="text-sm text-gray-600">
                                <span id="progress-personal">65%</span> / 
                                <span id="progress-benchmark" class="text-red-500">70%</span>
                            </span>
                        </div>
                        <div class="comparison-bar">
                            <div class="personal" style="width: 65%"></div>
                            <div class="benchmark" style="left: 70%"></div>
                        </div>
                        <p class="text-xs text-gray-500 mt-1">Thấp hơn <span id="progress-diff">5%</span> so với <span class="view-label">cohort</span></p>
                    </div>

                    <!-- Engagement -->
                    <div>
                        <div class="flex justify-between mb-2">
                            <span class="font-medium">Mức độ tương tác</span>
                            <span class="text-sm text-gray-600">
                                <span id="engagement-personal">45%</span> / 
                                <span id="engagement-benchmark" class="text-red-500">60%</span>
                            </span>
                        </div>
                        <div class="comparison-bar">
                            <div class="personal" style="width: 45%"></div>
                            <div class="benchmark" style="left: 60%"></div>
                        </div>
                        <p class="text-xs text-gray-500 mt-1">Cần cải thiện <span id="engagement-diff">15%</span></p>
                    </div>

                    <!-- Assessment Score -->
                    <div>
                        <div class="flex justify-between mb-2">
                            <span class="font-medium">Điểm kiểm tra TB</span>
                            <span class="text-sm text-gray-600">
                                <span id="score-personal">78%</span> / 
                                <span id="score-benchmark" class="text-green-500">75%</span>
                            </span>
                        </div>
                        <div class="comparison-bar">
                            <div class="personal" style="width: 78%"></div>
                            <div class="benchmark" style="left: 75%"></div>
                        </div>
                        <p class="text-xs text-gray-500 mt-1">Cao hơn <span id="score-diff">3%</span> 🎉</p>
                    </div>

                    <!-- Time Spent -->
                    <div>
                        <div class="flex justify-between mb-2">
                            <span class="font-medium">Thời gian học/tuần</span>
                            <span class="text-sm text-gray-600">
                                <span id="time-personal">8h</span> / 
                                <span id="time-benchmark" class="text-red-500">10h</span>
                            </span>
                        </div>
                        <div class="comparison-bar">
                            <div class="personal" style="width: 80%"></div>
                            <div class="benchmark" style="left: 100%"></div>
                        </div>
                        <p class="text-xs text-gray-500 mt-1">Thấp hơn <span id="time-diff">2h</span></p>
                    </div>
                </div>
            </div>

            <!-- Performance Trends -->
            <div class="bg-white rounded-lg shadow-sm p-6">
                <h2 class="text-xl font-bold mb-4">📈 Xu hướng Performance</h2>
                <canvas id="trendChart" height="300"></canvas>
            </div>

            <!-- Cohort Distribution -->
            <div class="bg-white rounded-lg shadow-sm p-6">
                <h2 class="text-xl font-bold mb-4">👥 Phân bố trong Cohort</h2>
                <canvas id="distributionChart" height="300"></canvas>
            </div>

            <!-- Risk Factors -->
            <div class="bg-white rounded-lg shadow-sm p-6">
                <h2 class="text-xl font-bold mb-4">⚠️ Yếu tố rủi ro</h2>
                <div class="space-y-3">
                    <div class="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                        <div>
                            <p class="font-medium text-red-800">Không hoạt động lâu</p>
                            <p class="text-sm text-red-600">5 ngày chưa học</p>
                        </div>
                        <span class="text-2xl">🚨</span>
                    </div>
                    <div class="flex items-center justify-between p-3 bg-orange-50 rounded-lg">
                        <div>
                            <p class="font-medium text-orange-800">Tương tác thấp</p>
                            <p class="text-sm text-orange-600">Dưới TB cohort 15%</p>
                        </div>
                        <span class="text-2xl">⚡</span>
                    </div>
                    <div class="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                        <div>
                            <p class="font-medium text-green-800">Điểm kiểm tra tốt</p>
                            <p class="text-sm text-green-600">Protective factor</p>
                        </div>
                        <span class="text-2xl">✅</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Hybrid Recommendations -->
        <div class="bg-white rounded-lg shadow-sm p-6 mt-6">
            <h2 class="text-xl font-bold mb-4">🎯 Khuyến nghị Hybrid (Cá nhân + Tổng thể)</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <!-- Recommendation 1 -->
                <div class="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                    <div class="flex items-center justify-between mb-2">
                        <span class="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">Ưu tiên cao</span>
                        <span class="text-sm text-gray-500">From: Risk Analysis</span>
                    </div>
                    <h3 class="font-bold mb-2">Cần hỗ trợ khẩn cấp</h3>
                    <p class="text-sm text-gray-600 mb-3">Risk level Medium, cần can thiệp để tránh dropout</p>
                    <div class="space-y-1">
                        <p class="text-xs text-gray-700">✓ Book 1-1 với mentor</p>
                        <p class="text-xs text-gray-700">✓ Join study group cohort</p>
                    </div>
                </div>

                <!-- Recommendation 2 -->
                <div class="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                    <div class="flex items-center justify-between mb-2">
                        <span class="px-2 py-1 bg-orange-100 text-orange-800 text-xs rounded-full">Quan trọng</span>
                        <span class="text-sm text-gray-500">From: Cohort Insights</span>
                    </div>
                    <h3 class="font-bold mb-2">Tăng cường tương tác</h3>
                    <p class="text-sm text-gray-600 mb-3">45% vs 60% cohort average</p>
                    <div class="space-y-1">
                        <p class="text-xs text-gray-700">✓ 1 forum post/tuần</p>
                        <p class="text-xs text-gray-700">✓ Join live sessions</p>
                    </div>
                </div>

                <!-- Recommendation 3 -->
                <div class="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                    <div class="flex items-center justify-between mb-2">
                        <span class="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">Cá nhân hóa</span>
                        <span class="text-sm text-gray-500">From: Personal Data</span>
                    </div>
                    <h3 class="font-bold mb-2">Học theo lịch cố định</h3>
                    <p class="text-sm text-gray-600 mb-3">Dựa trên pattern cá nhân</p>
                    <div class="space-y-1">
                        <p class="text-xs text-gray-700">✓ Tối 19-21h (peak time)</p>
                        <p class="text-xs text-gray-700">✓ 3 buổi/tuần minimum</p>
                    </div>
                </div>

                <!-- Recommendation 4 -->
                <div class="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                    <div class="flex items-center justify-between mb-2">
                        <span class="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">Leverage</span>
                        <span class="text-sm text-gray-500">From: Strengths</span>
                    </div>
                    <h3 class="font-bold mb-2">Tận dụng điểm mạnh</h3>
                    <p class="text-sm text-gray-600 mb-3">Điểm test cao hơn average</p>
                    <div class="space-y-1">
                        <p class="text-xs text-gray-700">✓ Peer tutoring</p>
                        <p class="text-xs text-gray-700">✓ Advanced content</p>
                    </div>
                </div>

                <!-- Recommendation 5 -->
                <div class="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                    <div class="flex items-center justify-between mb-2">
                        <span class="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">Content</span>
                        <span class="text-sm text-gray-500">From: Global Trends</span>
                    </div>
                    <h3 class="font-bold mb-2">Trending trong cohort</h3>
                    <p class="text-sm text-gray-600 mb-3">Popular với learners tương tự</p>
                    <div class="space-y-1">
                        <p class="text-xs text-gray-700">✓ Practical Projects</p>
                        <p class="text-xs text-gray-700">✓ Code Review Skills</p>
                    </div>
                </div>

                <!-- Recommendation 6 -->
                <div class="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                    <div class="flex items-center justify-between mb-2">
                        <span class="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded-full">System</span>
                        <span class="text-sm text-gray-500">From: AI Prediction</span>
                    </div>
                    <h3 class="font-bold mb-2">Adaptive Learning Path</h3>
                    <p class="text-sm text-gray-600 mb-3">Tối ưu cho normal pace learner</p>
                    <div class="space-y-1">
                        <p class="text-xs text-gray-700">✓ Adjust difficulty</p>
                        <p class="text-xs text-gray-700">✓ Personalized pace</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- System-wide Insights -->
        <div class="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 mt-6">
            <h2 class="text-xl font-bold mb-4">🌍 Insights từ Phân tích Tổng thể</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-white rounded-lg p-4">
                    <h3 class="font-semibold text-gray-700 mb-2">Peak Learning Hours</h3>
                    <p class="text-2xl font-bold text-blue-600">19-21h</p>
                    <p class="text-sm text-gray-600">80% học viên active</p>
                </div>
                <div class="bg-white rounded-lg p-4">
                    <h3 class="font-semibold text-gray-700 mb-2">Dropout Risk Pattern</h3>
                    <p class="text-2xl font-bold text-orange-600">Day 7</p>
                    <p class="text-sm text-gray-600">Critical intervention point</p>
                </div>
                <div class="bg-white rounded-lg p-4">
                    <h3 class="font-semibold text-gray-700 mb-2">Success Factor #1</h3>
                    <p class="text-2xl font-bold text-green-600">Consistency</p>
                    <p class="text-sm text-gray-600">3+ sessions/week = 85% completion</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data structures
        const benchmarkData = {
            cohort: {
                progress: 70,
                engagement: 60,
                score: 75,
                time: 10
            },
            global: {
                progress: 65,
                engagement: 55,
                score: 70,
                time: 8
            }
        };

        const personalData = {
            progress: 65,
            engagement: 45,
            score: 78,
            time: 8
        };

        let currentView = 'cohort';

        // Switch view function
        function switchView(view) {
            currentView = view;
            
            // Update buttons
            document.querySelectorAll('button[id^="btn-"]').forEach(btn => {
                btn.classList.remove('bg-blue-500', 'text-white');
                btn.classList.add('bg-gray-200', 'text-gray-700');
            });
            document.getElementById(`btn-${view === 'personal' ? 'personal' : view}`).classList.remove('bg-gray-200', 'text-gray-700');
            document.getElementById(`btn-${view === 'personal' ? 'personal' : view}`).classList.add('bg-blue-500', 'text-white');
            
            // Update comparisons
            if (view === 'personal') {
                // Hide benchmarks in personal view
                document.querySelectorAll('.benchmark').forEach(el => el.style.display = 'none');
                document.querySelectorAll('.view-label').forEach(el => el.textContent = 'mục tiêu cá nhân');
            } else {
                document.querySelectorAll('.benchmark').forEach(el => el.style.display = 'block');
                const benchmark = benchmarkData[view];
                
                // Update benchmark values
                document.getElementById('progress-benchmark').textContent = benchmark.progress + '%';
                document.getElementById('engagement-benchmark').textContent = benchmark.engagement + '%';
                document.getElementById('score-benchmark').textContent = benchmark.score + '%';
                document.getElementById('time-benchmark').textContent = benchmark.time + 'h';
                
                // Update benchmark positions
                document.querySelectorAll('.comparison-bar').forEach((bar, idx) => {
                    const benchmarkEl = bar.querySelector('.benchmark');
                    const values = [benchmark.progress, benchmark.engagement, benchmark.score, benchmark.time * 10];
                    benchmarkEl.style.left = values[idx] + '%';
                });
                
                // Update differences
                document.getElementById('progress-diff').textContent = Math.abs(personalData.progress - benchmark.progress) + '%';
                document.getElementById('engagement-diff').textContent = Math.abs(personalData.engagement - benchmark.engagement) + '%';
                document.getElementById('score-diff').textContent = Math.abs(personalData.score - benchmark.score) + '%';
                document.getElementById('time-diff').textContent = Math.abs(personalData.time - benchmark.time) + 'h';
                
                document.querySelectorAll('.view-label').forEach(el => el.textContent = view);
            }
            
            // Update charts
            updateCharts();
        }

        // Initialize charts
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        const trendChart = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
                datasets: [{
                    label: 'Personal',
                    data: [45, 52, 58, 62, 60, 65],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.3
                }, {
                    label: 'Cohort Avg',
                    data: [50, 55, 60, 65, 68, 70],
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        const distCtx = document.getElementById('distributionChart').getContext('2d');
        const distChart = new Chart(distCtx, {
            type: 'bar',
            data: {
                labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                datasets: [{
                    label: 'Cohort Distribution',
                    data: [5, 15, 35, 30, 15],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.5)',
                        'rgba(251, 146, 60, 0.5)',
                        'rgba(250, 204, 21, 0.5)',
                        'rgba(34, 197, 94, 0.5)',
                        'rgba(59, 130, 246, 0.5)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                xMin: 3.2,
                                xMax: 3.2,
                                borderColor: 'rgb(59, 130, 246)',
                                borderWidth: 3,
                                label: {
                                    content: 'You',
                                    enabled: true,
                                    position: 'start'
                                }
                            }
                        }
                    }
                }
            }
        });

        function updateCharts() {
            // Update trend chart based on view
            if (currentView === 'global') {
                trendChart.data.datasets[1].data = [45, 50, 55, 60, 63, 65];
                trendChart.data.datasets[1].label = 'Global Avg';
            } else {
                trendChart.data.datasets[1].data = [50, 55, 60, 65, 68, 70];
                trendChart.data.datasets[1].label = 'Cohort Avg';
            }
            trendChart.update();
        }

        // Initialize
        switchView('cohort');
    </script>
</body>
</html>