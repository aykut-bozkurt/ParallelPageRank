run_tests () {
	echo Successfully compiled.

	# run tests
	echo running ranking with static scheduling with chunk size of 100
	./a.out 100 static
	echo Finished \#1

	echo running ranking with static scheduling with chunk size of 200
	./a.out 200 static
	echo Finished \#2

	echo running ranking with static scheduling with chunk size of 300
	./a.out 300 static
	echo Finished \#3

	echo running ranking with static scheduling with chunk size of 400
	./a.out 400 static
	echo Finished \#4

	echo running ranking with dynamic scheduling with chunk size of 100
	./a.out 100 dynamic
	echo Finished \#5

	echo running ranking with dynamic scheduling with chunk size of 200
	./a.out 200 dynamic
	echo Finished \#6

	echo running ranking with dynamic scheduling with chunk size of 300
	./a.out 300 dynamic
	echo Finished \#7

	echo running ranking with dynamic scheduling with chunk size of 400
	./a.out 400 dynamic
	echo Finished \#8
}



# compile
echo Compiling...
(g++ p_ranking.c -fopenmp) && run_tests

