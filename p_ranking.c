#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <time.h>
#include <queue>

#define TOKEN_LENGTH 26
 
//#define N 5
#define ALPHA 0.8
#define EPSILON 0.000001

using namespace std;

typedef struct CSRMatrix matrix;

struct CSRMatrix {
   int n_nodes;
   int n_edges;
   int * rowstarts;
   int * colindices;
   float * values;
   vector<string> *node_list; 
};

matrix * constructCSRMatrix(FILE *fptr);

void normalize(matrix *P);

// ALGORITHM => r(t+1) = alpha*P*r(t) + (1-alpha)*c ; c = r(0) = [1,...,1] ; alpha = 0.2
 int main() {
	 
    matrix *P;
    
    
    
    int i, j, k, step;
    float totalDiff;
	int N;
    // read matrix file
    FILE *fptr;
    fptr = fopen("../graph.txt","r");
    if(fptr == NULL)
    {
            printf("File not found!");
            exit(1);
    }

    P = constructCSRMatrix(fptr);
	N = P->n_nodes;
    fclose(fptr);
    float * r = (float*)calloc(N, sizeof(float));
    float * nextR = (float*)calloc(N, sizeof(float));
    // normalize P matrix values (each row sum = 1.0 total probability)
    normalize(P);
	
	printf("Normalized");

    /* Some initializations */
    for (i=0; i<N; i++) {
        r[i] = 1.0/N;
    }
	
	/*for (i=0; i<P->n_edges; i++) {
       printf("valu: %.6f \n ", P->values[i]);
    }*/

    time_t t;
    srand((unsigned) time(&t));
    float t0,t1;
    step = 1;
    
    while(1){
        #pragma omp parallel shared(P, r, nextR, N	) private(i, j, k)
        {
            t0 = omp_get_wtime();
			
            // calculate nextR
            #pragma omp for
            for(i=0; i<N; i++){
            	for(k=P->rowstarts[i]; k<P->rowstarts[i+1]; k++){
                	nextR[i] += ALPHA*P->values[k]*r[P->colindices[k]];
                }
				//printf("Next R at i: %f \n ", nextR[i]);
                nextR[i] += (1-ALPHA)/N;
				//printf("Next R at i: %f \n ", nextR[i]);
            }

			// calculate difference to compare with epsilon
            #pragma omp for reduction (+:totalDiff)
			for(i=0; i<N; i++){
                totalDiff += fabs(r[i]-nextR[i]);
            }

            t1 = omp_get_wtime();
            printf("Thread%d spent %f secs in the parallel region.\n", omp_get_thread_num(), t1-t0);
        } // end of parallel section
        
        
        printf("Step: %d\n", step);
        printf("Difference: %.6f\n", totalDiff);
        if(totalDiff <= EPSILON) break;
        
    // update iteration variables

		totalDiff = 0;	
		for(i=0; i<N; i++){
			r[i] = nextR[i];
			nextR[i] = 0;
		}
        step++;
    }
		
	ofstream myfile;
	myfile.open("ranks.txt");
	
    printf("Resultant Ranks\n");
	for(i=0; i<N ; i++){
		//printf("ranks[%d]=%.10f\n", i, r[i]);
		myfile << "ranks[" << (*(P->node_list))[i] << "]=" << r[i] << endl;
		
	}
    
	// Find 5 largest ranks
	priority_queue<pair<float, string>, vector< pair<float, string> > ,greater< pair<float, string> > > q; 
	
	for(int i=0; i<N; i++) {
		float rank = r[i];
		string name = (*(P->node_list))[i];
		
		if(i<5){
			q.push(make_pair(rank,name));
		}
		else {
			float smallest = q.top().first;
			if(rank>smallest) {
				q.pop();
				q.push(make_pair(rank,name));
			}
		}
	}
	
	printf("5 largest elements: \n");
	for (int i=0; i<5; i++){
		pair<float,string> front = q.top();
		float value = front.first;
		string name = front.second;
		
		q.pop();
		
		//printf("Name %s, Rank %.10f \n", name,value);
		cout << "Name " << name << " Rank " << value << endl;
	}
	
    
    // free unused heap memory
    /*free(P.rowstarts);
    free(P.colindices);
    free(P.values);*/
    free(r);
    free(nextR);
 }

matrix * constructCSRMatrix(FILE *fp){
	
    map<string,int> index_map;
	map<int, vector<int>* > adj_map;
	
	int first_index, second_index;
	int last_index = 0;
	vector<int>* neighbours;
	vector<string>* node_list = new vector<string>();
	char ch;
	
	char first_node[TOKEN_LENGTH+1];
	char second_node[TOKEN_LENGTH+1];
	
	int steps = 0;
	
	ch = fgetc(fp);

	while (ch!=EOF) {
		first_node[0] = ch;
		for (int i=1; i<TOKEN_LENGTH; i++){
			ch = fgetc(fp);
			first_node[i] = ch;
		}
		
		first_node[TOKEN_LENGTH] = '\0';
		
		ch = fgetc(fp);	// read empty delimiter
		
		for (int i=0; i<TOKEN_LENGTH; i++){
			ch = fgetc(fp);
			second_node[i] = ch;
		}
		second_node[TOKEN_LENGTH] = '\0';
		
		ch = fgetc(fp); // read newline
		ch = fgetc(fp); // read next character or EOF
		
		try {
			first_index = index_map.at(first_node); 
		}
		catch (exception e){
			
			index_map.insert(pair<string, int>(first_node, last_index));
			
			first_index = last_index;
			node_list->push_back(first_node);
			last_index += 1;
		}
		
		try {
			second_index = index_map.at(second_node); 
		}
		catch (exception e){
			
			index_map.insert(pair<string, int>(second_node, last_index));
			
			second_index = last_index;
			node_list->push_back(second_node);
			last_index += 1;
		}
		
		try {
			neighbours = adj_map.at(second_index); 
		}
		catch (exception e){
			neighbours = new vector<int>();
			adj_map.insert(make_pair(second_index, neighbours));
		}
		neighbours->push_back(first_index);
		
		
		
		
		steps += 1;

		//break;
	}
	
	int *rowstarts = new int[last_index+1];
	int  *colindices = new int[steps];
	float *values = new float[steps];
	
	int last_row_idx = 0;
	rowstarts[0] = last_row_idx;
	
	for (int i=0; i<last_index; i++){
		try {
			neighbours = adj_map.at(i);
			
			for (int j=0; j<neighbours->size(); j++) {
				colindices[last_row_idx+j] = (*neighbours)[j];
				values[last_row_idx+j] = 1;
			}
			
			last_row_idx += neighbours->size();
			rowstarts[i+1] = last_row_idx;
			
		}
		catch (exception e){
			rowstarts[i+1] = last_row_idx;
		}
	}
	
	matrix * P = (matrix*)malloc(1*sizeof(matrix));
	P->colindices = colindices;
	P->rowstarts = rowstarts;
	P->values = values;
	P->n_edges = steps;
	P->n_nodes = last_index;
	P->node_list = node_list;
	printf("Number of nodes %d", P->n_nodes);
	
    return P;
}

void normalize(matrix * P){
	
	int i,k;
    
    float *colsum = new float[P->n_nodes];
    printf("Normalizing");
	printf("Number of nodes %d", P->n_nodes);
	for(k=0;k<P->n_nodes;k++){
		for(i=P->rowstarts[k]; i<P->rowstarts[k+1]; i++){
			colsum[P->colindices[i]] += P->values[i];
		}
	}	
    for(k=0;k<P->n_nodes;k++){
		for(i=P->rowstarts[k]; i<P->rowstarts[k+1]; i++){
			P->values[i] /= colsum[P->colindices[i]];
		}
	}

}

