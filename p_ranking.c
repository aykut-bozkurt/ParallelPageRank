#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
 
#define N 5
#define ALPHA 0.2
#define BETA 0.8
#define EPSILON 0.000001

int readinputs(FILE *fptr, int *rowstarts, int *values, int *colindices);

float ** getP(int rowstartssize, int *rowstarts, int *values, int *colindices);

void normalize(float ** P);

// ALGORITHM
// r(t+1) = 0.2*P*r(t) + 0.8*c   ; c = r(0) = r 
// r1 = 0.2*(P*r) + 0.8*(r)     r2 = 0.2^2*(P^2*r) + 0.2*0.8*(P*r) + 0.8*(r)   r3 = 0.2^3*(P^3*r) + 0.2^2*0.8*(P^2*r) + 0.2*0.8*(P*r) + 0.8*(r)
// 1. diff(r2 - r1) = [ 0.2^2*(P^2*r) - 0.2*(P*r) ] + 0.2*0.8*(P*r)
// 2. diff(r3 - r2) = [ 0.2^3*(P^3*r) - 0.2^2*(P^2*r) ] + 0.2^2*0.8*(P^2*r)   

 int main() {

 	float ** P;
 	float * r0 = (float*)malloc(N * sizeof(float));
	float ** resultRank = (float**)malloc(N * sizeof(float*));

 	float ** prevP = (float**)malloc(N * sizeof(float*)); // a^1 * P^1 at first
 	float ** nextP = (float**)malloc(N * sizeof(float*)); // 0 at first
	float ** alphaBetaP = (float**)malloc(N * sizeof(float*)); // 0.2 * 0.8 * P at first
 	float totalDiff = 0;
 	int i, j, k, step;

	// read matrix file
	FILE *fptr;
	fptr = fopen("matrix.txt","r");
	if(fptr == NULL)
   	{
      		printf("File not found!");
      		exit(1);
   	}

	int * rowstarts = (int*) malloc(25*sizeof(int));
	int * values = (int*) malloc(25*sizeof(int));
	int * colindices = (int*) malloc(25*sizeof(int));
	k = readinputs(fptr, rowstarts, values, colindices);
	fclose(fptr);
	
	// construct P matrix
	P = getP(k, rowstarts, values, colindices);
	normalize(P);
	

 	/* Some initializations */
 	for (i=0; i < N; i++) {
   		r0[i] = 1.0;
 	}
	for (i=0; i < N; i++) {
		//P[i] = (float*)malloc(N * sizeof(float));
		alphaBetaP[i] = (float*)malloc(N * sizeof(float));
		prevP[i] = (float*)malloc(N * sizeof(float));
		nextP[i] = (float*)malloc(N * sizeof(float));
		resultRank[i] = (float*)malloc(N * sizeof(float));
   		for (j=0; j < N; j++) {
			//P[i][j] = 0.1 * ALPHA; // take this from file.txt
			alphaBetaP[i][j] = ALPHA * BETA * P[i][j];
			prevP[i][j] = P[i][j];
   			nextP[i][j] = 0;
			resultRank[i][j] = 0;
 		}
 	}

	time_t t;
	srand((unsigned) time(&t));
	float t0,t1;
 	step = 1;
	while(1){
		#pragma omp parallel shared(P, r0, prevP, nextP, alphaBetaP, resultRank, t) private(i, j, k, t0, t1)
		{
			t0 = omp_get_wtime();
   			// calculate nextP
   			#pragma omp for
			 for(i=0; i<N; i++){
				 for(j=0; j<N ; j++){
					 for(k=0; k<N ; k++){
						 nextP[i][j] = nextP[i][j] + ALPHA * P[i][k] * prevP[k][j];
					 }
				 }
			 }

			// calculate totalDiff
			#pragma omp for reduction(+:totalDiff)
			 for(i=0; i<N; i++){
				for(k=0; k<N ; k++){
					resultRank[i][i] += nextP[i][k]*r0[k] - prevP[i][k]*r0[k] + alphaBetaP[i][k]*r0[k];
					totalDiff +=  fabs( nextP[i][k]*r0[k] - prevP[i][k]*r0[k] + alphaBetaP[i][k]*r0[k] );
				}
		        }

			// update alphaBetaP
			#pragma omp for
			 for(i=0; i<N; i++){
				for(k=0; k<N ; k++){
					alphaBetaP[i][k] = 0.2 * P[i][k] * alphaBetaP[i][k];
			 	}
			 }

			t1 = omp_get_wtime();
			printf("Thread%d spent %f secs in the parallel region.\n", omp_get_thread_num(), t1-t0);
   		} // end of parallel section

		printf("Step: %d\n", step);
		printf("%.6f\n", totalDiff);

   		if(totalDiff <= EPSILON) break;
		
		// free unused heap memory
		for(i=0; i<N; i++){
			free(prevP[i]);
		}
		
		// update iteration variables
   		prevP = nextP;
		totalDiff = 0;
   		step++;
	}

	printf("Resultant Ranks\n");
	for(i=0; i<N ; i++){
		printf("ranks[%d]=%.6f\n", i, resultRank[i][i] + r0[i]);
	}

	// free unused heap memory
	for(i=0; i<N; i++){
		free(P[i]);
		free(prevP[i]); // nextP points to same address
		free(alphaBetaP[i]);
		free(resultRank[i]);
	}
	free(P);
	free(prevP);
	free(alphaBetaP);
	free(resultRank);
	free(r0);

 }

int readinputs(FILE *fptr, int *rowstarts, int *values, int *colindices){
	char *token;
	const char* delim = " ";
	char * line;
	size_t len = 0;
    	ssize_t read = 0;
	
	int k = 0;
	int l = 0;
	int m = 0;
	int linecount = 0;
	int tokencount = 0;
	while((read = getline(&line, &len, fptr)) != -1){
		printf("%s", line);
		token = strtok(line, delim);
		linecount++;
		while(token != NULL){
			if(strcmp(token, "[") != 0 && strcmp(token, "]") != 0 && strcmp(token, "row_begin") != 0 && strcmp(token, "col_indices") != 0 && strcmp(token, "values") != 0 && strcmp(token, "=") != 0) {
				if(linecount == 1){
					rowstarts[k] = atoi(token);
					k++;
				}
				else if(linecount == 2){
					values[l] = atoi(token);
					l++;
				}
				else if(linecount == 3){
					colindices[m] = atoi(token);
					m++;
				}	
			}
			token = strtok(NULL, " ");
		}
	}

	if(line != NULL) free(line);

	return k;
}

float ** getP(int rowstartssize, int *rowstarts, int *values, int *colindices){
	int j = 0;
	int t = 0;
	float ** P = (float**) malloc(N*sizeof(float*));
	for(j=0; j<N ; j++){
		P[j] = (float*) calloc(N, sizeof(float));
	}
	int row;
	for(row=0; row<rowstartssize-2 ; row++){
		int elemcount = rowstarts[row+1] - rowstarts[row];
		for(j=t; j<t+elemcount; j++){
			P[row][colindices[j]] = values[j];
		}
		t = t + elemcount;
	}

	for(j=0; j<N ; j++){
		for(t=0; t<N ; t++){
			printf("%.f ", P[j][t]);
		}
		printf("\n");
	}

	return P;
}

void normalize(float ** P){
	int i,j;
	for(i=0 ; i<N ; i++){
		float rowsum = 0;
		for(j=0 ; j < N ; j++){
			rowsum += P[i][j];
		}
		for(j=0 ; j < N ; j++){
			P[i][j] = P[i][j]/rowsum;
		}
	}
}
