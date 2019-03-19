run_tests () {
	echo Successfully compiled.

	# run tests
	:' echo running ranking with static scheduling with chunk size of 100
	./a.out 1 100 static
	echo Finished \#1

	echo running ranking with static scheduling with chunk size of 200
	./a.out 2 200 static
	echo Finished \#2

	echo running ranking with dynamic scheduling with chunk size of 100
	./a.out 3 100 dynamic
	echo Finished \#3

	echo running ranking with dynamic scheduling with chunk size of 200
	./a.out 4 200 dynamic
	echo Finished \#4 '

	echo running ranking with guided scheduling with minimum chunk size of 50
	./a.out 5 50 guided
	echo Finished \#1

	echo running ranking with guided scheduling with minimum chunk size of 10
	./a.out 6 10 guided
	echo Finished \#2
}



# compile
echo Compiling...
(g++ p_ranking.c -fopenmp) && run_tests

