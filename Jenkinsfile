// =============================================================
// Jenkinsfile（备查——CI/CD 示意流水线）
//
// 阶段：
//   1. 单元测试（全 mock，不依赖外部服务，快）
//   2. 构建镜像
//   3. （可选）评估：起服务 → 跑 evaluate.py → 留存报告
//
// 【设计点：为什么评估不放进每次 CI（面试点）】
// 评估跑 15 条用例 = 15 次 Agent 请求 × 8~12 次 LLM 调用，
// 每次 CI 都跑成本高（token 费用）且耗时长。
// 合理策略：单元测试每次 PR 必跑（秒级），完整评估
// 在版本发布 / 模型切换 / prompt 大改时手动触发或定时跑——
// 模型行为没变时，重跑评估没有信息增量。
// =============================================================

pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Unit Tests') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'cd backend && python -m pytest tests/ -v'
            }
        }

        stage('Build Image') {
            steps {
                sh 'docker build -f docker/Dockerfile -t enterprise-agent:latest .'
            }
        }

        stage('Evaluation (manual)') {
            when {
                // 评估耗 token，仅在手动触发时执行（见文件头注释）
                expression { params.RUN_EVALUATION == true }
            }
            steps {
                sh '''
                    docker compose -f docker/docker-compose.yml up -d --build
                    sleep 30
                    docker compose -f docker/docker-compose.yml exec app python -m app.database.seed
                    python evaluation/evaluate.py --api-url http://localhost:8000
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'evaluation/evaluation_report.json', allowEmptyArchive: true
                }
            }
        }
    }

    parameters {
        booleanParam(name: 'RUN_EVALUATION', defaultValue: false,
                     description: '是否运行完整 Agent 评估（消耗 LLM token）')
    }
}
