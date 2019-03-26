run_tests () {
	echo Successfully compiled.

	# run tests
	export OMP_SCHEDULE="static, 100"	
	echo running ranking with static scheduling with chunk size of 100
	./a.out 1 100 static
	echo Finished \#1

	export OMP_SCHEDULE="static, 200"
	echo running ranking with static scheduling with chunk size of 200
	./a.out 2 200 static
	echo Finished \#2

	export OMP_SCHEDULE="static, 500"	
	echo running ranking with static scheduling with chunk size of 500
	./a.out 3 500 static
	echo Finished \#3

	export OMP_SCHEDULE="dynamic, 100"
	echo running ranking with dynamic scheduling with chunk size of 100
	./a.out 4 100 dynamic
	echo Finished \#4

	export OMP_SCHEDULE="dynamic, 200"
	echo running ranking with dynamic scheduling with chunk size of 200
	./a.out 5 200 dynamic
	echo Finished \#5 

	export OMP_SCHEDULE="dynamic, 500"
	echo running ranking with dynamic scheduling with chunk size of 500
	./a.out 6 500 dynamic
	echo Finished \#6

	export OMP_SCHEDULE="guided, 500"
	echo running ranking with guided scheduling with minimum chunk size of 500
	./a.out 7 500 guided
	echo Finished \#7

	export OMP_SCHEDULE="guided, 50"
	echo running ranking with guided scheduling with minimum chunk size of 50
	./a.out 8 50 guided
	echo Finished \#8

	export OMP_SCHEDULE="guided, 10"
	echo running ranking with guided scheduling with minimum chunk size of 10
	./a.out 9 10 guided
	echo Finished \#9
}



# compile
echo Compiling...
(g++ p_ranking.c -fopenmp) && run_tests

