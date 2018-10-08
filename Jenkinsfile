pipeline {
  agent none

  stages {

    stage("check") {
      agent  {
          docker {
            image 'dealii/dealii:v8.5.1-gcc-mpi-fulldepscandi-debugrelease'
              label 'has-docker'
              args '-v /home/docker/jenkins:/home/dealii/jenkins'
              }
      }

      // skip this if it is not a pull request:
      when { expression { env.CHANGE_ID != null } }

      steps {
        githubNotify context: 'CI', description: 'need ready to test label and /rebuild',  status: 'PENDING'

        // For /rebuild to work you need to:
        // 1) select "issue comment" to be delivered in the github webhook setting
        // 2) install "GitHub PR Comment Build Plugin" on Jenkins
        // 3) in project settings select "add property" "Trigger build on pr comment" with
        //    the phrase ".*/rebuild.*" (without quotes)
        sh '''
            wget -q -O - https://api.github.com/repos/tjhei/cracks/issues/${CHANGE_ID}/labels | grep 'ready to test' || \
            { echo "This commit will only be tested when it has the label 'ready to test'. Trigger a rebuild by adding a comment that contains '/rebuild'..."; exit 1; }
        '''
        githubNotify context: 'CI', description: 'running tests...',  status: 'PENDING'
      }
    }

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
          stage("indent") {
            steps {
                sh './contrib/indent'
                sh 'git diff > changes.diff'
                archiveArtifacts artifacts: 'changes.diff', fingerprint: true
                sh '''
                    git diff --exit-code || \
                    { echo "Please check indentation!"; exit 1; }
                '''
                githubNotify context: 'indent', description: '',  status: 'SUCCESS'
            }
          }

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
            image 'dealii/dealii:v9.0.0-gcc-mpi-fulldepscandi-debugrelease'
            label 'has-docker'
            args '-v /home/docker/jenkins:/home/dealii/jenkins'
          }
        }

        stages {
            stage("build") {
                steps {
                    sh 'cmake -D CMAKE_CXX_FLAGS="-Werror" .'
                    sh 'make -j 4'
                }
            }

            stage('test') {
                steps {
                    sh 'export OMPI_MCA_btl=self,tcp;./cracks'
                    sh 'export OMPI_MCA_btl=self,tcp;ctest --output-on-failure -j 4 || { touch FAILED; cat tests/output-*/diff >test.diff; } '
                    archiveArtifacts artifacts: 'test.diff', fingerprint: true, allowEmptyArchive: true
                    sh 'export OMPI_MCA_btl=self,tcp;ctest --output-on-failure -j 4'
                    sh 'if [ -f FAILED ]; then exit 1; fi'
                }
            }

            stage("end") {
              steps {
                githubNotify context: 'CI', description: 'success!',  status: 'SUCCESS'
              }
            }
        }

        post { cleanup { cleanWs() } }
    }

  }

}
