#include <omp.h>

#include <math.h>

#include <stdio.h>

#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>

#include <exception>

#include <time.h>

#include <queue>
#include <map>
#include <vector>

 
#define ALPHA 0.8
#define EPSILON 0.000001

// ALGORITHM => r(t+1) = alpha*P*r(t) + (1-alpha)*c ; c = r(0) = [1,...,1] ; alpha = 0.2
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

matrix * constructCSRMatrix();

void normalize(matrix *P);

void printtop5rank(float *r, matrix *P, int N);

void write_ranks_to_file(float *r, matrix *P, int N);

 int main(int argc, char* argv[]) {
    int test_no = atoi(argv[1]);
    int chunk_size = atoi(argv[2]);
    string schedule_method = argv[3];

    stringstream csv_stream;
    csv_stream << test_no << ", ";
    csv_stream << schedule_method << ", ";
    csv_stream << chunk_size << ", ";

    matrix *P;
    
    int i, j, k, step;
    float totalDiff;
    int N;

    // read graph.txt and construct CSR matrix 
    P = constructCSRMatrix();

    // total number of vertices
    N = P->n_nodes;
   
    // current and next rank matrices
    float * r = (float*)calloc(N, sizeof(float));
    float * nextR = (float*)calloc(N, sizeof(float));

    // normalize P matrix values (each column sum = 1.0 total probability)
    normalize(P);


    float thread_timings[8];
    int tid;
    /* Some initializations */
    for (i=0; i<N; i++) {
        r[i] = 1.0/N;
    }
    for(i=0; i<8; i++){
	thread_timings[i] = 0;
    }

    time_t t;
    srand((unsigned) time(&t));
    float t0,t1;
    step = 1;
    while(1){
        #pragma omp parallel shared(P, r, nextR, N, chunk_size, schedule_method, thread_timings) private(i, j, k, t0, t1, tid) num_threads(8)
        {
            t0 = omp_get_wtime();
			
            // calculate nextR
            #pragma omp for schedule(dynamic,chunk_size)
            for(i=0; i<N; i++){
            	for(k=P->rowstarts[i]; k<P->rowstarts[i+1]; k++){
                	nextR[i] += ALPHA*P->values[k]*r[P->colindices[k]];
                }
                nextR[i] += (1-ALPHA)/N;
            }

	    // calculate difference to compare with epsilon
            #pragma omp for reduction (+:totalDiff) schedule(dynamic,chunk_size)
	    for(i=0; i<N; i++){
                totalDiff += fabs(r[i]-nextR[i]);
            }

            t1 = omp_get_wtime();
	    tid = omp_get_thread_num();
	    thread_timings[tid] += (t1-t0);
            printf("Thread%d spent %f secs in the parallel region.\n", tid, thread_timings[tid]);
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

    csv_stream << step << ", |";
    for(int i=0; i<8; i++){
	csv_stream << thread_timings[i] << "|";
    }
    csv_stream << endl;
	
    // output top 5 ranks
    printtop5rank(r, P, N);
	
    // write all ranks to rank.txt
    write_ranks_to_file(r, P, N);

    // write test results to csv file
    ofstream csvfile;
    csvfile.open ("testresults.csv", ios_base::app);
    if(test_no == 1) {
	csvfile << "Test No, Scheduling Method, Chunk Size, No. of iterations, Timings in secs for each number of threads" << endl;
	csvfile << ",,,, |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |" << endl;
    }
    csvfile << csv_stream.str();
    csvfile.close();

    // free unused heap memory
    free(P->rowstarts);
    free(P->colindices);
    free(P->values);
    free(P);
    free(r);
    free(nextR);
 }

matrix * constructCSRMatrix(){
	map<string,int> index_map;
	map<int, vector<int>* > adj_map;
	
	int first_index, second_index;
	int last_index = 0;
	vector<int>* neighbours;
	vector<string>* node_list = new vector<string>();

	ifstream myfile ("graph.txt");

	int steps = 0;
	string first_node(26,'0');
	string second_node(26,'0');
	string token(26,'0');
	while(!myfile.eof())
	{
    		for(int i=0;i<2;i++)
    		{
			myfile >> token;
        		if(i == 0) {
				first_node = token;
			}
			else {
				second_node = token;
			}
    		}

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
	}

	myfile.close();

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
	
	return P;
}

void normalize(matrix * P){
    int i,k;
    
    float *colsum = new float[P->n_nodes];
	
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

void printtop5rank(float* r, matrix *P, int N){
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
    
    // prints top 5 rank	
    printf("5 top rank vertices: \n");
    for (int i=0; i<5; i++){
	pair<float,string> front = q.top();
	float value = front.first;
	string name = front.second;

	q.pop();
		
	cout << "Name " << name << " Rank " << value << endl;
    }
}

void write_ranks_to_file(float *r, matrix *P, int N){
    // write ranks to file
    ofstream myfile;
    myfile.open("ranks.txt");
	
    for(int i=0; i<N ; i++){
    	myfile << "ranks[" << (*(P->node_list))[i] << "]=" << r[i] << endl;	
    }
}

