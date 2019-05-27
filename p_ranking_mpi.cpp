#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <exception>
#include <chrono>
#include <math.h>

#include <queue>
#include <map>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>

#include <mpi.h>
#include <metis.h>


 
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

matrix * constructCSRMatrix(char* filename);

void normalize(matrix *P);

void printtop5rank(float *r, matrix *P, int N);

void write_ranks_to_file(float *r, matrix *P, int N);

int* getPartition(matrix* P, int nparts);
int* calculateOffsets(int size, const int* counts);
tuple<matrix*,int*,int*,int*> splitMatrix(matrix* P, int nparts, int* partition);

int TMPI_Scatterv_buffered(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype,
                 int root, MPI_Comm comm, int buffer_size);

int main(int argc, char* argv[]) {

	char* filename = argv[1];

	int mypid;
	int numprocs;
	

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

	int* nodes;
	int* node_counts; 
	int* edge_counts;
	int* node_offsets; 
	int* edge_offsets;
	int total_nodes;
	int total_edges;
	int* partition;
	float* global_ranks;
	float* old_global_ranks;


	matrix* Q = new matrix;
	matrix* P = new matrix;

	int nnodes, nedges;

	if(mypid==0) {  
		// read graph from disk
		cout << "Reading graph from disk." << endl;
		P = constructCSRMatrix(filename);

		normalize(P);

		// partition the graph into N subgraphs
		cout << "Computing graph partition." << endl;
		partition = getPartition(P, numprocs);


		cout << "Compute matrix splits." << endl;

		/* We re-order arrays of the matrix in accordance to 
		the partition so that they can be easily send with Scatterv.*/
		tuple<matrix*,int*,int*,int*> results  = splitMatrix(P, numprocs, partition);
		Q = get<0>(results);
		nodes = get<1>(results);
		node_counts = get<2>(results);
		edge_counts = get<3>(results);

		cout << "Calculate offsets." << endl;
		node_offsets = calculateOffsets(numprocs, node_counts);
		edge_offsets = calculateOffsets(numprocs, edge_counts);


		total_nodes = Q->n_nodes;
		cout << total_nodes << endl;
		for (int i=0; i<numprocs; i++) {
			cout << "Node count: " << node_counts[i] << " Edge count: " << \
				edge_counts[i] << " Node offset: " << node_offsets[i] << " Edge offset: " << edge_offsets[i] << endl;
		}

		int max_col_idx = 0;
		for (int i=0; i<P->n_edges; i++){
			int idx = P->colindices[i];
			if(idx>max_col_idx)
				max_col_idx = idx;
		}
		cout << "Max col idx: " << max_col_idx << endl;

		max_col_idx = 0;
		for (int i=0; i<Q->n_edges; i++){
			int idx = Q->colindices[i];
			if(idx>max_col_idx)
				max_col_idx = idx;
		}
		cout << "Max col idx: " << max_col_idx << endl;

		cout << "nedges: "  << Q->n_edges << endl;

		total_edges = Q->n_edges;

		int _sum = 0;
		for (int i = 0; i < numprocs; ++i)
		{
			_sum += edge_counts[i];
		}
		cout << "sum of edges: "<< _sum << endl;

		global_ranks = new float[total_nodes];
		old_global_ranks = new float[total_nodes];

		for (int i = 0; i < total_nodes; ++i)
		{
			old_global_ranks[i] = 1/total_nodes;
			global_ranks[i] = 1/total_nodes;
		}
	}	

	MPI_Bcast(&total_nodes, 1 , MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&total_edges, 1 , MPI_INT, 0, MPI_COMM_WORLD);

	if(mypid>0) {
		partition = new int[total_nodes];
	}

	MPI_Bcast(partition, total_nodes, MPI_INT, 0, MPI_COMM_WORLD);

	int edge_start;

	MPI_Scatter(node_counts, 1, MPI_INT, &nnodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(edge_counts, 1, MPI_INT, &nedges, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(edge_offsets, 1, MPI_INT, &edge_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	cout << "mypid=" << mypid << ", partition size=" << nnodes << ",  nedges=" << nedges << endl;

	int* local_nodes = new int[nnodes];
	int* rowstarts = new int[nnodes];
	int* colindices = new int[nedges];
	float* values = new float[nedges];
	
	/* Because OpenMPI Scatter function seems to hanging for large message sizes, 
	we send the messages in chucks given by buffer size. */
	int buffer_size = 64000;

	TMPI_Scatterv_buffered(nodes, node_counts, node_offsets, MPI_INT, local_nodes, nnodes, MPI_INT, 0, MPI_COMM_WORLD, buffer_size);
	TMPI_Scatterv_buffered(Q->rowstarts, node_counts, node_offsets, MPI_INT, rowstarts, nnodes, MPI_INT, 0, MPI_COMM_WORLD, buffer_size);
	TMPI_Scatterv_buffered(Q->colindices, edge_counts, edge_offsets, MPI_INT, colindices, nedges, MPI_INT, 0, MPI_COMM_WORLD, buffer_size);
	TMPI_Scatterv_buffered(Q->values, edge_counts, edge_offsets, MPI_FLOAT, values, nedges, MPI_FLOAT, 0, MPI_COMM_WORLD, buffer_size);

	// MPI_Scatterv(nodes, node_counts, node_offsets, MPI_INT, local_nodes, nnodes, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Scatterv(Q->rowstarts, node_counts, node_offsets, MPI_INT, rowstarts, nnodes, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Scatterv(Q->colindices, edge_counts, edge_offsets, MPI_INT, colindices, nedges, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Scatterv(Q->values, edge_counts, edge_offsets, MPI_FLOAT, values, nedges, MPI_FLOAT, 0, MPI_COMM_WORLD);

		
	// Re-order nodes & find neighbors

	unordered_map<int,int> index_map;
	float* send_buffer;
	float* recv_buffer;
	int send_cnts[numprocs];
	int recv_cnts[numprocs];
	int send_offsets[numprocs];
	int recv_offsets[numprocs];
	set<int> send_map[numprocs];
	set<int> recv_map[numprocs];

	/* We re-index nodes occurring in the colindices array 
	so that nodes that are in our partition come before the 
	foreign nodes. We replace the new local index values with 
	the old ones to speed up main loop. */

	cout << "Initialize local arrays..." << endl;
	for(int i=0; i<nnodes; i++) {
		rowstarts[i] -= edge_start; 
		index_map[local_nodes[i]] = i;
	}

	int last_col_idx = 0;
	int last_index = nnodes;
	int local_idx, index;


	for(int i=0; i<nnodes;i++) {
		for(int j=last_col_idx; j<rowstarts[i]; j++) {
			index = colindices[j];
			
			if(!index_map.count(index)) {
				index_map[index] = last_index;
				last_index++;
			}

			local_idx = index_map[index];
			
			if(local_idx>=nnodes) {
				int part = partition[index];
				send_map[part].insert(i);
				recv_map[part].insert(index);
			}
			
			colindices[j] = local_idx;
		}
		last_col_idx = rowstarts[i];
	}

	/* Each processor calculates which nodes it needs from 
	every other partition and which nodes it needs to send.
	Note that ranks will be put in the order of global ranks
	so that node order of send buffer of the sending processor
	matches the order of send buffer of the sending processor. */

	cout << "Create send-recieve maps, mypid " << mypid << endl;

	int total_send = 0;
	int total_recv = 0;
	for (int i = 0; i < numprocs; ++i) {
		send_cnts[i] = send_map[i].size();
		recv_cnts[i] = recv_map[i].size();

		total_send += send_map[i].size();
		total_recv += recv_map[i].size();

		if(i>0) {
			send_offsets[i] = send_offsets[i-1] + send_map[i-1].size();
			recv_offsets[i] = recv_offsets[i-1] + recv_map[i-1].size();
		} else {
			send_offsets[i] = 0;
			recv_offsets[i] = 0;
		}
	}

	send_buffer = new float[total_send];
	recv_buffer = new float[total_recv];

	// initialize local rank arrays
	float* ranks = new float[nnodes]();
	float* old_ranks = new float[nnodes + total_recv];


	for (int i = 0; i < nnodes + total_recv; ++i)
	{
		old_ranks[i] = 1.0/total_nodes;
	}

	if(mypid==0) {
		global_ranks = new float[total_nodes];
		old_global_ranks = new float[total_nodes];

		for(int i=0; i<total_nodes; i++) {
			old_global_ranks[i] = 1.0 / total_nodes;
		}
	}

	cout << "id: " << mypid << " last_index: " << last_index << " total_recv " << total_recv << " total_send " << total_send<< " nnodes " << nnodes << endl;
	
	// Main Iteration

	float prob;
	int node_idx;
	int converged = 0;
	int iter_cnt = 0;

	MPI_Barrier(MPI_COMM_WORLD);

	auto start = std::chrono::steady_clock::now( );


	while(!converged && iter_cnt<100) {
		// Fill send buffer	
		for (int i=0; i<numprocs; i++) {
			int j = send_offsets[i];
			for (auto local_idx : send_map[i]) {
				send_buffer[j] = old_ranks[local_idx];
				j++;
			}
		}

		//MPI_Barrier(MPI_COMM_WORLD);
		MPI_Alltoallv(send_buffer, send_cnts, send_offsets, MPI_FLOAT, recv_buffer, recv_cnts, recv_offsets, MPI_FLOAT, MPI_COMM_WORLD);

		// Put ranks recieved from other nodes to correct places
		for (int i=0; i<numprocs; i++) {
			int j = recv_offsets[i];
			for (auto index : recv_map[i]) {
				local_idx = index_map[index];
				old_ranks[local_idx] = recv_buffer[j];
				j++;
			}
		}

		int last_col_idx = 0;
		for(int i=0; i<nnodes;i++) {
			for(int j=last_col_idx; j<rowstarts[i]; j++) {
            	ranks[i] += ALPHA*values[j]*old_ranks[colindices[j]];;
            }
            ranks[i] += (1-ALPHA)/total_nodes;

            last_col_idx = rowstarts[i];
        }

        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gatherv(ranks, nnodes, MPI_FLOAT, global_ranks, node_counts, node_offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (mypid==0) {
        	cout << "Iteration: " << iter_cnt << endl; 
        	iter_cnt ++;
        	
        	float totalDiff = 0;
        	float sum = 0;

        	for(int i=0; i<total_nodes; i++){
        		sum += global_ranks[i];
                totalDiff += fabs(global_ranks[i]-old_global_ranks[i]);
            }
            cout << "totalDiff: " << totalDiff << endl;
            cout << "sum of ranks: " << sum << endl;
            if(totalDiff <= EPSILON) {
            	converged = 1;
            }
        }
        MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&iter_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < nnodes; ++i)
		{
			old_ranks[i] = ranks[i];
			ranks[i] = 0;
		}

		if(mypid==0) {
			for (int i = 0; i < total_nodes; ++i)
			{
				old_global_ranks[i] = global_ranks[i];
				global_ranks[i] = 0;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - start );
    cout << "Operation took " << elapsed.count() << " millisecs on processor	" << mypid << endl;

	MPI_Finalize();

	if(mypid==0) {
		for (int i = 0; i < total_nodes; ++i)
		{
			global_ranks[nodes[i]] = old_global_ranks[i];
		}
		
		// print vertices with highest rank
		printtop5rank(global_ranks, P, total_nodes);

	}
	return 0;
 }

matrix * constructCSRMatrix(char* filename){
	unordered_map<string,int> index_map;
	unordered_map<int, unordered_set<int>*> adj_map;
	unordered_set<int>* neighbours;
	vector<string>* node_list = new vector<string>();

	int first_index, second_index;
	int last_index = 0;
	
	ifstream myfile (filename);

	int steps = 0;
	
	string first_node;
	string second_node;

	while(myfile >> first_node >> second_node)
	{
		//cout << first_node << " " << second_node << endl;

		try {
			first_index = index_map.at(first_node); 
		}
		catch (exception e){
			
			index_map.insert(pair<string, int>(first_node, last_index));
			
			first_index = last_index;
			node_list->push_back(first_node);
			last_index += 1;

			neighbours = new unordered_set<int>();
			adj_map.insert(make_pair(first_index, neighbours));
		}
		
		try {
			second_index = index_map.at(second_node); 
		}
		catch (exception e){
			
			index_map.insert(pair<string, int>(second_node, last_index));
			
			second_index = last_index;
			node_list->push_back(second_node);
			last_index += 1;

			neighbours = new unordered_set<int>();
			adj_map.insert(make_pair(second_index, neighbours));
		}
		
		
		neighbours = adj_map.at(second_index);
		if (!neighbours->count(first_index))
			steps += 1;
		neighbours->insert(first_index);


		neighbours = adj_map.at(first_index);
		if (!neighbours->count(second_index))
			steps += 1;
		neighbours->insert(second_index);
		
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
			int j = 0;
			for (auto neighbour : *neighbours) {
				colindices[last_row_idx+j] = neighbour;
				values[last_row_idx+j] = 1;
				j++;
			}
			
			last_row_idx += j;
			rowstarts[i+1] = last_row_idx;
			
		}
		catch (exception e){
			rowstarts[i+1] = last_row_idx;
		}
	}

	matrix * P = new matrix;
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
    
    float *colsum = new float[P->n_nodes]();
	
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
    priority_queue<pair<float, string>, vector< pair<float, string> >, greater< pair<float, string> > > q; 
    
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
    
    cout << endl;
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

int* getPartition(matrix* P, int nparts) {
	idx_t nvertices = P->n_nodes;
	idx_t nedges = P->n_edges;
	idx_t ncon = 1;

	idx_t* xadj = P->rowstarts;
	idx_t* adjacency = P->colindices;

	idx_t objval;
	idx_t* part = new idx_t[nvertices];

	METIS_PartGraphKway(&nvertices, &ncon, xadj, adjacency, NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, part);

	return part;
}

/* Re-order arrays of the matrix so that the in all arrays 
values corresponding to nodes from the first partition occur 
first, then the values for the second partition come after 
those for the first partition etc. After this operation we 
can easily send the arrays using Scatterv. */ 
tuple<matrix*,int*, int*,int*> splitMatrix(matrix* P, int nparts, int* partition) {
	int nvertices = P->n_nodes;
	int nedges = P->n_edges;
	int* rowstarts = P->rowstarts;
	int* colindices = P->colindices;
	float* values = P->values;
	vector<string> *node_list = P->node_list;

	int* new_rowstarts = new int[nvertices]();
	int* new_colindices = new int[nedges]();
	int* new_nodes = new int[nvertices]();
	float* new_values = new float[nedges]();
	vector<string> *new_node_list;
	new_node_list->reserve(nvertices);


	int* node_counts = new int[nparts]();
	int* edge_counts = new int[nparts]();

	vector<pair<int,int>> order;
	for (int i=0; i<nvertices; i++) {
		order.push_back(make_pair(partition[i], i));
	}
	sort(order.begin(), order.end());

	int last_col_idx = 0;
	for (int i=0; i<nvertices; i++) {
		int part = order[i].first;
		int orig_idx = order[i].second;

		new_nodes[i] = orig_idx;

		int nneighbours = rowstarts[orig_idx+1] - rowstarts[orig_idx];
		node_counts[part] += 1;
		edge_counts[part] += nneighbours;

		for (int j=0; j<nneighbours; j++) {
			int new_idx = last_col_idx + j;
			int old_idx = rowstarts[orig_idx] + j;
			new_colindices[new_idx] = colindices[old_idx];
			new_values[new_idx] = values[old_idx];
		}

		last_col_idx += nneighbours;
		new_rowstarts[i] = last_col_idx;
	}

	cout << "Initialize new matrix." << endl;

    matrix * Q = new matrix;
	Q->colindices = new_colindices;
	Q->rowstarts = new_rowstarts;
	Q->values = new_values;
	Q->n_edges = nedges;
	Q->n_nodes = nvertices;
	Q->node_list = new_node_list;

	return make_tuple(Q, new_nodes, node_counts, edge_counts);
}

int* calculateOffsets(int size, const int* counts) {
	int * offsets = new int[size];
	int sum = 0;
	for (int i = 0; i < size; ++i)
	{
		offsets[i] = sum;
		sum += counts[i];
	}
	return offsets;
}

/* Break a Scatterv call into several calls so that the 
number elements send from the root to one vertex is always 
smaller or equal to buffer size. */
int TMPI_Scatterv_buffered(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype,
                 int root, MPI_Comm comm, int buffer_size) 
{

	int numprocs, mypid;

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

	int datatype_size;
  	MPI_Type_size(sendtype, &datatype_size);
	
	char* recv_pointer = (char*) recvbuf;
	int recv_size;
	int remaining = recvcount;

	int recv_sizes[numprocs];
	int total_remaining = 1;
	int offsets[numprocs];

	if(mypid==root) {
		total_remaining = 0;
		for (int i = 0; i < numprocs; ++i) {
			total_remaining += sendcounts[i];
			offsets[i] = displs[i];
		}
	}

	MPI_Bcast(&total_remaining, 1, MPI_INT, root, MPI_COMM_WORLD);
	
	while(total_remaining>0)
	{
		recv_size = min(remaining, buffer_size);
		remaining -= recv_size;

		MPI_Gather(&recv_size, 1, sendtype, recv_sizes, 1, sendtype, root, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Scatterv(sendbuf, recv_sizes, offsets, sendtype, recv_pointer, recv_size, sendtype, root, MPI_COMM_WORLD);

		recv_pointer += recv_size*datatype_size;

		if(mypid==root) {
			int total_transfer = 0;
			for (int i = 0; i < numprocs; ++i)
			{	
				int transfer_size = recv_sizes[i];
				offsets[i] += transfer_size;
				total_transfer += transfer_size;
			}
			total_remaining -= total_transfer;
		}
		MPI_Bcast(&total_remaining, 1, MPI_INT, root, MPI_COMM_WORLD);
	}

	return 0;
}