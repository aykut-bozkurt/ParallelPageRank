#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
 
#define N 5
#define ALPHA 0.2
#define EPSILON 0.000001

typedef struct CSRMatrix matrix;

struct CSRMatrix {
   int n_rows;
   int n_cols;
   int * rowstarts;
   int * colindices;
   float * values;
};

int readinputs(FILE *fptr, int *rowstarts, float *values, int *colindices);

void normalize(matrix P);

// ALGORITHM => r(t+1) = alpha*P*r(t) + (1-alpha)*c ; c = r(0) = [1,...,1] ; alpha = 0.2
 int main() {

    matrix P;
    P.rowstarts = (int*)calloc(N+1, sizeof(int));
    P.colindices = (int*)calloc(N*N, sizeof(int));
    P.values = (float*)calloc(N*N, sizeof(float));
    
    float * r = (float*)calloc(N, sizeof(float));
    float * nextR = (float*)calloc(N, sizeof(float));
    
    int i, j, k, step;
    float totalDiff;

    // read matrix file
    FILE *fptr;
    fptr = fopen("matrix.txt","r");
    if(fptr == NULL)
    {
            printf("File not found!");
            exit(1);
    }

    k = readinputs(fptr, P.rowstarts, P.values, P.colindices);
    fclose(fptr);
    
    // normalize P matrix values (each row sum = 1.0 total probability)
    normalize(P);

    /* Some initializations */
    for (i=0; i<N; i++) {
        r[i] = 1.0/N;
    }
    

    time_t t;
    srand((unsigned) time(&t));
    float t0,t1;
    step = 1;
    
    while(1){
        #pragma omp parallel shared(P, r, nextR) private(i, j, k)
        {
            t0 = omp_get_wtime();

            // calculate nextR
            #pragma omp for
            for(i=0; i<N; i++){
            	for(k=P.rowstarts[i]; k<P.rowstarts[i+1]; k++){
                	nextR[i] += ALPHA*P.values[k]*r[P.colindices[k]];
                }

                nextR[i] += (1-ALPHA)/N;
            }

            

            t1 = omp_get_wtime();
            printf("Thread%d spent %f secs in the parallel region.\n", omp_get_thread_num(), t1-t0);
        } // end of parallel section
        
        // calculate difference to compare with epsilon
            for(i=0; i<N; i++){
                totalDiff += fabs(r[i]-nextR[i]);
            }
        printf("Step: %d\n", step);
        printf("Difference: %.6f\n", totalDiff);
        if(totalDiff <= EPSILON) break;
        
    // update iteration variables

    totalDiff = 0;
	#pragma omp for	
	for(i=0; i<N; i++){
		r[i] = nextR[i];
        nextR[i] = 0;
	}
        step++;
    }

    printf("Resultant Ranks\n");
    for(i=0; i<N ; i++){
        printf("ranks[%d]=%.6f\n", i, nextR[i]);
    }
    
    
    // free unused heap memory
    /*free(P.rowstarts);
    free(P.colindices);
    free(P.values);*/
    free(r);
    free(nextR);
 }

int readinputs(FILE *fptr, int *rowstarts, float *values, int *colindices){
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
                    values[l] = (float) atoi(token);
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

void normalize(matrix P){
	int i,k;
    
    int colsum[N];
    
	for(k=0;k<N;k++){
		for(i=P.rowstarts[k]; i<P.rowstarts[k+1]; i++){
			colsum[P.colindices[i]] += P.values[i];
		}
	}	
    for(k=0;k<N;k++){
		for(i=P.rowstarts[k]; i<P.rowstarts[k+1]; i++){
			P.values[i] /= colsum[P.colindices[i]];
		}
	}

}

