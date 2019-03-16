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

struct CSRMatrix {
   int n_rows;
   int n_cols;
   int * rowstarts;
   int * colindices;
   float * values;
};

int readinputs(FILE *fptr, int *rowstarts, float *values, int *colindices);

float ** getP(int rowstartssize, int *rowstarts, int *values, int *colindices);

void normalize(float ** P);

// ALGORITHM
// r(t+1) = 0.2*P*r(t) + 0.8*c   ; c = r(0) = r 
// r1 = 0.2*(P*r) + 0.8*(r)     r2 = 0.2^2*(P^2*r) + 0.2*0.8*(P*r) + 0.8*(r)   r3 = 0.2^3*(P^3*r) + 0.2^2*0.8*(P^2*r) + 0.2*0.8*(P*r) + 0.8*(r)
// 1. diff(r2 - r1) = [ 0.2^2*(P^2*r) - 0.2*(P*r) ] + 0.2*0.8*(P*r)
// 2. diff(r3 - r2) = [ 0.2^3*(P^3*r) - 0.2^2*(P^2*r) ] + 0.2^2*0.8*(P^2*r)   

 int main() {

    struct CSRMatrix P;
    //printf("Here");
    P.rowstarts = (int*)calloc(N+1, sizeof(int));
    P.colindices = (int*)calloc(N*N, sizeof(int));
    P.values = (float*)calloc(N*N, sizeof(float));
    
    float * r = (float*)calloc(N, sizeof(float));
    float * nextR = (float*)calloc(N, sizeof(float));
    
    int i, j, k, step;
    float totalDiff = 0;

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
    
    // construct P matrix
    //P = getP(k, rowstarts, values, colindices);
    //normalize(P);
    

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
            // calculate nextP
            #pragma omp for
             for(i=0; i<N; i++){
                 for(k=P.rowstarts[i]; k<P.rowstarts[i+1]; k++){
                     nextR[i] = nextR[i] + ALPHA*P.values[k]*r[P.colindices[k]];
                 }
                 //printf("%0.6f \n", nextR[i]);
                 nextR[i] = nextR[i] + (1-ALPHA)/N;
                 //printf("%0.6f \n ", (1-ALPHA)/N);
             }

            // calculate totalDiff
            #pragma omp for reduction(+:totalDiff)
            for(i=0; i<N; i++){
                totalDiff +=  fabs( nextR[i] - r[i] );
            }

            t1 = omp_get_wtime();
            printf("Thread%d spent %f secs in the parallel region.\n", omp_get_thread_num(), t1-t0);
        } // end of parallel section

        printf("Step: %d\n", step);
        printf("%.6f\n", totalDiff);
        //break;

        if(totalDiff <= EPSILON) break;
        
        // free unused heap memory
        free(r);
        
        // update iteration variables
        r = nextR;
        totalDiff = 0;
        step++;
    }

    printf("Resultant Ranks\n");
    for(i=0; i<N ; i++){
        printf("ranks[%d]=%.6f\n", i, nextR[i]);
    }
    
    // free unused heap memory
    //free(P.rowstarts);
    //free(P.colindices);
    //free(P.values);
    free(r);
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

void normalize(float ** P){
    int i,j;
    for(i=0 ; i<N ; i++){
        float rowsum = 0;
        for(j=0 ; j < N ; j++){
            rowsum += P[i][j];
        }
        for(j=0 ; j < N ; j++){
            P[i][j] = ( 0.1*rowsum/N + 0.9*P[i][j] )/rowsum;
        }
    }
}
