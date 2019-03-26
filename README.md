ParallelPageRank


1) Execute below test generate test results

#> bash run.sh 

2) Set schedule method via environment variable

You should set environment variable to change schedule method before compilation phase.

E.G 
export OMP_SCHEDULE="static, 100"
export OMP_SCHEDULE="static, 500"
export OMP_SCHEDULE="guided, 50"

3) Compile to run standalone by giving command line arguments

#> g++ -fopenmp p_ranking.c <testid> <chunksize> <schedulemethod>

Commandline arguments:
tesid: integer
chunksize: integer
schedulemethod: static | dynamic | guided 

Note: Commandline arguments are used to generate csv file

