pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "project-150342d7-e272-4d0c-ab9"
        IMAGE_NAME = "ml-project"
    }

    stages {
        stage('Cloning GitHub repo to Jenkins') {
            steps {
                checkout scmGit(
                    branches: [[name: '*/main']],
                    extensions: [],
                    userRemoteConfigs: [[
                        credentialsId: 'github-token',
                        url: 'https://github.com/SafwenCherif/hotel-reservation-mlops.git'
                    ]]
                )
            }
        }

        stage('Setting up Virtual Environment and Installing dependencies') {
            steps {
                sh '''
                python3 -m venv ${VENV_DIR}
                . ${VENV_DIR}/bin/activate
                pip install --upgrade pip
                pip install -e .
                '''
            }
        }

        stage('Running Training Pipeline (with ADC mounted)') {
            steps {
                sh '''
                . ${VENV_DIR}/bin/activate
                export PATH=$PATH:/usr/bin
                export GOOGLE_APPLICATION_CREDENTIALS="/home/jenkins/.config/gcloud/application_default_credentials.json"
                export GOOGLE_CLOUD_PROJECT="${GCP_PROJECT}"
                python pipeline/training_pipeline.py
                '''
            }
        }

        stage('Building and Pushing Docker Image to GCR') {
            steps {
                script {
                    sh '''
                    export PATH=$PATH:/usr/bin

                    gcloud config set project ${GCP_PROJECT}
                    gcloud auth configure-docker --quiet

                    docker build -t gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest .
                    docker push gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest
                    '''
                }
            }
        }

        stage('Deploy to Google Cloud Run') {
            steps {
                script {
                    sh '''
                    export PATH=$PATH:/usr/bin

                    gcloud config set project ${GCP_PROJECT}

                    gcloud run deploy ${IMAGE_NAME} \
                        --image=gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest \
                        --platform=managed \
                        --region=us-central1 \
                        --allow-unauthenticated
                    '''
                }
            }
        }
    }
}
