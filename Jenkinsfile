pipeline {
  agent  { 
    docker {
      image 'dealii/dealii:v8.5.1-gcc-mpi-fulldepscandi-debugrelease' /*dealii/base:gcc-mpi'*/
	label 'has-docker'
	args '-v /home/docker/jenkins:/home/dealii/jenkins'
	}
  }  

  stages
    {

    stage("conf") {
      steps {
	echo "Running build ${env.BUILD_ID} on ${env.NODE_NAME}, env=${env.NODE_ENV}"
	  sh 'printenv'
	  sh 'ls -al'
	  sh 'cmake .'
	  }
    }
    stage("build") {
      steps {
      	sh 'make -j 4'                
	  }
    }
    stage('Test') {
      steps {
	echo 'Testing..'
	sh './cracks'
	  }
    }
  }
  
}
