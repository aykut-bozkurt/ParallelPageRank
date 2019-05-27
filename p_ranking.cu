#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <math.h>

#include <stdio.h>

#include <stdlib.h>

#include <iostream>
#include <fstream>

#include <string>

#include <exception>

#include <time.h>

#include <queue>
#include <map>
#include <vector>

 
#define ALPHA 0.2
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


struct diffabs_functor {

	diffabs_functor(){}

	__host__ __device__
		float operator()(const float& x, const float& y) const {
			return abs(x-y);
		}
};

// x = how many nonzero values arethere in current row
// y = how many nonzero values arethere in the next row
struct matrix_vec_mul_functor {
	float alpha;
	thrust::device_ptr<float> rx;
	thrust::device_ptr<int> colindicesx;
	thrust::device_ptr<float> valuesx;
	
	matrix_vec_mul_functor(float _alpha, thrust::device_vector<float> &r_x, thrust::device_vector<int> &colindices_x, thrust::device_vector<float> &values_x) : alpha(_alpha) {
		rx = &r_x[0];
		colindicesx = &colindices_x[0];
		valuesx = &values_x[0];
	}
	__host__ __device__
		float operator()(const int& x, const int& y) const {
			int i;
			float sum = 0;
			for(i = x; i < y; i++){
				sum += valuesx[i] * rx[colindicesx[i]];
			}
			return alpha * sum;
		}
};


matrix * constructCSRMatrix();

void normalize(matrix *P);

void printtop5rank(float *r, matrix *P, int N);

 int main(int argc, char* argv[]) {
    // transition probabilities matrix
    matrix *P;
    
    // iteration variables
    int i, step;

    // difference that is compared to epsilon at each iteration
    float totalDiff = 0;
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

    /* Some initializations */
    for (i=0; i<N; i++) {
        r[i] = 1.0/N;
    }
	
    // transfer P related matrices to device
    thrust::device_vector<int> colindices_x(P->colindices, P->colindices+P->rowstarts[N]);
    thrust::device_vector<float> values_x(P->values, P->values+P->rowstarts[N]);
    thrust::device_vector<int> rowstarts_x(P->rowstarts, P->rowstarts+N+1);
	
    // transfer r and nextR to device
    thrust::device_vector<float> r_x(r, r+N);
    thrust::device_vector<float> nextR_x(nextR, nextR+N);

    // create other operational matrices in device
    thrust::device_vector<float> diffabs(N);
    thrust::device_vector<float> one_minus_alpha_over_n(N);
    thrust::device_vector<float> row_col_mul(N);
	
    time_t start,end;
    time (&start);
    step = 1;
    printf("Total nonzero value is %d ",P->rowstarts[N]);
    while(1){
	thrust::fill(diffabs.begin(), diffabs.end(), 0);

        // calculate nextR = alpha*P*r_x + (1-alpha)*r_x (P in CSR format)
	thrust::transform(rowstarts_x.begin(), rowstarts_x.end()-1, rowstarts_x.begin()+1, nextR_x.begin(), matrix_vec_mul_functor(ALPHA, r_x, colindices_x, values_x));
       
	thrust::fill(one_minus_alpha_over_n.begin(), one_minus_alpha_over_n.end(), (1-ALPHA)/N);
	thrust::transform(nextR_x.begin(), nextR_x.end(), one_minus_alpha_over_n.begin(), nextR_x.begin(), thrust::plus<float>());
	
	// calculate totaldiff in this step
	thrust::transform(r_x.begin(), r_x.end(), nextR_x.begin(), diffabs.begin(), diffabs_functor());
	totalDiff = thrust::reduce(diffabs.begin(), diffabs.end(), (float)0, thrust::plus<float>());
		
        printf("Step: %d\n", step);
        printf("Difference: %.6f\n", totalDiff);
        if(totalDiff <= EPSILON) break;
		
	// update r_x and nextR_x
	thrust::copy_n(nextR_x.begin(), N, r_x.begin());
	thrust::fill(nextR_x.begin(), nextR_x.end(), 0);
        
	// update iteration variables
	totalDiff = 0;	
	step++;
    } // end of while (algorithm converged)
	
    time (&end);
    printf("Operation took %.9f secs on GPU.\n", difftime (end,start));

    // transfer to host
    thrust::copy(r_x.begin(), r_x.end(), r);

    // output top 5 ranks
    printtop5rank(r, P, N);

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
    
    float *colsum = (float*)calloc(P->n_nodes, sizeof(float));
	
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
