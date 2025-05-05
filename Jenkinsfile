pipeline {
  agent none

  stages {

    stage("check") {
      agent  {
          docker {
            image 'dealii/dealii:v9.6.0-jammy'
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

    stage ("9.6") {
        agent  {
          docker {
            image 'dealii/dealii:v9.6.0-jammy'
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
                    sh './cracks'
                }
            }
        }

        post { cleanup { cleanWs() } }
    }

    stage ("9.5") {
        agent  {
          docker {
            image 'dealii/dealii:v9.5.1-jammy'
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
                    sh 'cmake -D CMAKE_CXX_FLAGS="-Werror" .'
                    sh 'make -j 4'
                }
          }

          stage('test') {
                steps {
                    sh './cracks'
                    sh 'ctest --output-on-failure || { touch FAILED; cat tests/output-*/*diff >test.diff; } '
                    archiveArtifacts artifacts: 'test.diff', fingerprint: true, allowEmptyArchive: true
                    sh 'ctest --output-on-failure'
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
