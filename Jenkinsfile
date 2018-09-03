pipeline {
  agent none

  stages {


    stage("main") {
        agent  {
          docker {
            image 'dealii/dealii:v8.5.1-gcc-mpi-fulldepscandi-debugrelease'
              label 'has-docker'
              args '-v /home/docker/jenkins:/home/dealii/jenkins'
              }
        }

        stages {
            stage("conf") {
                steps {
                    echo "Running build ${env.BUILD_ID} on ${env.NODE_NAME}, env=${env.NODE_ENV}"
                    sh 'printenv'
                    sh 'ls -al'
                }
            }

            stage("indent") {
                steps {
                    sh './contrib/indent'
                    sh 'git diff > changes.diff'
                    archiveArtifacts artifacts: 'changes.diff', fingerprint: true
                    sh '''
                        git diff --exit-code || \
                        { echo "Please check indentation!"; exit 1; }
                    '''
                }
            }
        }
	post { cleanup { cleanWs() } }
    }


    stage ("8.5") {
        agent  {
          docker {
            image 'dealii/dealii:v8.5.1-gcc-mpi-fulldepscandi-debugrelease'
            label 'has-docker'
            args '-v /home/docker/jenkins:/home/dealii/jenkins'
          }
        }

        stages {

            stage("build") {
                steps {
                    sh 'cmake .'
                    sh 'make -j 4'
                }
            }

            stage('test') {
                steps {
                    sh './cracks'
                }
            }
        }
	post { cleanup { cleanWs() } }
    }


    stage ("9.0") {
        agent  {
          docker {
            image 'dealii/dealii:9.0.0-gcc-mpi-fulldepsspack-debugrelease'
            label 'has-docker'
            args '-v /home/docker/jenkins:/home/dealii/jenkins'
          }
        }

        stages {
            stage("build") {
                steps {
                    sh '''bash -c ". /etc/profile.d/spack.sh && \
                       cmake . && \
                       make -j 4"
                       '''
                }
            }

            stage('test') {
                steps {
                    sh 'echo "disabled for now ./cracks"'
                }
            }
        }
	post { cleanup { cleanWs() } }
    }

  }

}
