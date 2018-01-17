// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>

#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <algorithm>
//#include <iomanip>

typedef float WeightT;
struct Edge {
	int dst;
	WeightT wt;
};

bool compare_id(Edge a, Edge b) { return (a.dst < b.dst); }

void fill_data_block(int m, int &nnz, int *&row_offsets, int *&column_indices, WeightT *&weight, vector<vector<Edge> > vertices, bool symmetrize, bool sorted, bool remove_selfloops, bool remove_redundents) {
	//sort the neighbor list
	if(sorted) {
		printf("Sorting the neighbor lists...");
		for(int i = 0; i < m; i++) {
			std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
		}
		printf(" Done\n");
	}

	//remove self loops
	int num_selfloops = 0;
	if(remove_selfloops) {
		printf("Removing self loops...");
		for(int i = 0; i < m; i++) {
			for(unsigned j = 0; j < vertices[i].size(); j ++) {
				if(i == vertices[i][j].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_selfloops ++;
					j --;
				}
			}
		}
		printf(" %d selfloops are removed\n", num_selfloops);
	}

	// remove redundent
	int num_redundents = 0;
	if(remove_redundents) {
		printf("Removing redundent edges...");
		for (int i = 0; i < m; i++) {
			for (unsigned j = 1; j < vertices[i].size(); j ++) {
				if (vertices[i][j].dst == vertices[i][j-1].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_redundents ++;
					j --;
				}
			}
		}
		printf(" %d redundent edges are removed\n", num_redundents);
	}

/*
	// print some neighbor lists
	for (int i = 0; i < 3; i++) {
		cout << "src " << i << ": ";
		for (int j = 0; j < vertices[i].size(); j ++)
			cout << vertices[i][j].dst << "  ";
		cout << endl;
	}
*/
	row_offsets = (int *)malloc((m + 1) * sizeof(int));
	int count = 0;
	for (int i = 0; i < m; i++) {
		row_offsets[i] = count;
		count += vertices[i].size();
	}
	row_offsets[m] = count;
	if (symmetrize) {
		if(count != nnz) {
			nnz = count;
		}
	} else {
		if (count + num_selfloops + num_redundents != nnz)
			printf("Error reading graph, number of edges in edge list %d != %d\n", count, nnz);
		nnz = count;
	}
	printf("num_vertices %d num_edges %d\n", m, nnz);
	/*
	double avgdeg;
	double variance = 0.0;
	int maxdeg = 0;
	int mindeg = m;
	avgdeg = (double)nnz / m;
	for (int i = 0; i < m; i++) {
		int deg_i = row_offsets[i + 1] - row_offsets[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / m;
	}
	printf("min_degree %d max_degree %d avg_degree %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	*/
	column_indices = (int *)malloc(count * sizeof(int));
	weight = (WeightT *)malloc(count * sizeof(WeightT));
	vector<Edge>::iterator neighbor_list;
	for (int i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index] = (*neighbor_list).dst;
			weight[index] = (*neighbor_list).wt;
			index ++;
			neighbor_list ++;
		}
	}
	
	//zy second version, not test yet
	vector<int> block_column;
	for(int i = 0; i < num_rows; i ++){
		num_block[i] = row_offsets[i+height] - row_offsets[i]
		for(int offset = row_offsets[i]; offset < row_offsets[i+height]; offset ++){
			block_column[i].push_back(column[offset]/width);
			for(int offset_1 = row_offsets[i]; offset_1 < offset; offset_1++){		
				if(column[offset] == column[offset_1]){
					block_column[i].erase(column[offset]/width);
					num_block[i]--;
				}
			}
		}
	}

	//zy fist version, not efficient, not test right or wrong
/*	vector<int> row_offsets_block;
	vector<int> column_block;
	row_offsets_block[0] = 0;
	for (int i = 1; i < (m+1)/2; i++) 
		row_offsets_block[i] = row_offsets_block[i-1] + row_offsets[2*i] - row_offsets[2*(i-1)];
	for (int i = 0; i < (m+1)/2; i++) {
		num = row_offsets[2*i+1] - row_offsets[2*i];
		if(num > 0){
			for (int offset = row_offsets[2*i]; offset < row_offsets[2*i+1]-1; offset++){ 		
				if(column[offset]/2 == column[offset+1]/2){
					row_offsets_block[i+1]--;
					column_block.push_back(column[offset]/2);
					offset ++;
				}
				else column_block.push_back(column[offset]/2);

			}
			for (int offset = row_offsets[2*i+1]; offset < row_offsets[2*i+2]-1; offset++){ 		
				if(column[offset]/2 == column[offset+1]/2){
					row_offsets_block[i+1]--;
					column_block.push_back(column[offset]/2);
					offset ++;
				}
				else column_block.push_back(column[offset]/2);

				for (int offset_1 = row_offsets[2*i]; offset_1 < row_offsets[2*i+1]-1; offset_1++){ 
					if(column[offset]/2 == column[offset_1]/2){
						row_offsets_block[i+1]--;
						column_block.erase(column[offset]/2);
					}
				}
			}
		}



		row_offsets_block.push_back(row_offsets[i]);
		if((i > 0) && (row_offsets[i] == row_offsets[i-1]))
			row_offsets_block.erase(row_offsets[i]);
		if((i%2) != 0)
			column_indices[m]
	}
*/	
	/*
	// print some neighbor lists
	for (int i = 0; i < 6; i++) {
		int row_begin = row_offsets[i];
		int row_end = row_offsets[i + 1];
		cout << "src " << i << ": ";
		for (int j = row_begin; j < row_end; j ++)
			cout << column_indices[j] << "  ";
		cout << endl;
	}
	//*/
	//for (int i = 0; i < 10; i++) cout << weight[i] << ", ";
	//cout << endl;
}




void fill_data(int m, int &nnz, int *&row_offsets, int *&column_indices, WeightT *&weight, vector<vector<Edge> > vertices, bool symmetrize, bool sorted, bool remove_selfloops, bool remove_redundents) {
	//sort the neighbor list
	if(sorted) {
		printf("Sorting the neighbor lists...");
		for(int i = 0; i < m; i++) {
			std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
		}
		printf(" Done\n");
	}

	//remove self loops
	int num_selfloops = 0;
	if(remove_selfloops) {
		printf("Removing self loops...");
		for(int i = 0; i < m; i++) {
			for(unsigned j = 0; j < vertices[i].size(); j ++) {
				if(i == vertices[i][j].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_selfloops ++;
					j --;
				}
			}
		}
		printf(" %d selfloops are removed\n", num_selfloops);
	}

	// remove redundent
	int num_redundents = 0;
	if(remove_redundents) {
		printf("Removing redundent edges...");
		for (int i = 0; i < m; i++) {
			for (unsigned j = 1; j < vertices[i].size(); j ++) {
				if (vertices[i][j].dst == vertices[i][j-1].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_redundents ++;
					j --;
				}
			}
		}
		printf(" %d redundent edges are removed\n", num_redundents);
	}

/*
	// print some neighbor lists
	for (int i = 0; i < 3; i++) {
		cout << "src " << i << ": ";
		for (int j = 0; j < vertices[i].size(); j ++)
			cout << vertices[i][j].dst << "  ";
		cout << endl;
	}
*/
	row_offsets = (int *)malloc((m + 1) * sizeof(int));
	int count = 0;
	for (int i = 0; i < m; i++) {
		row_offsets[i] = count;
		count += vertices[i].size();
	}
	row_offsets[m] = count;
	if (symmetrize) {
		if(count != nnz) {
			nnz = count;
		}
	} else {
		if (count + num_selfloops + num_redundents != nnz)
			printf("Error reading graph, number of edges in edge list %d != %d\n", count, nnz);
		nnz = count;
	}
	printf("num_vertices %d num_edges %d\n", m, nnz);
	/*
	double avgdeg;
	double variance = 0.0;
	int maxdeg = 0;
	int mindeg = m;
	avgdeg = (double)nnz / m;
	for (int i = 0; i < m; i++) {
		int deg_i = row_offsets[i + 1] - row_offsets[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / m;
	}
	printf("min_degree %d max_degree %d avg_degree %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	*/
	column_indices = (int *)malloc(count * sizeof(int));
	weight = (WeightT *)malloc(count * sizeof(WeightT));
	vector<Edge>::iterator neighbor_list;
	for (int i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index] = (*neighbor_list).dst;
			weight[index] = (*neighbor_list).wt;
			index ++;
			neighbor_list ++;
		}
	}
	/*
	// print some neighbor lists
	for (int i = 0; i < 6; i++) {
		int row_begin = row_offsets[i];
		int row_end = row_offsets[i + 1];
		cout << "src " << i << ": ";
		for (int j = row_begin; j < row_end; j ++)
			cout << column_indices[j] << "  ";
		cout << endl;
	}
	//*/
	//for (int i = 0; i < 10; i++) cout << weight[i] << ", ";
	//cout << endl;
}

// transfer R-MAT generated gr graph to CSR format
void gr2csr(char *gr, int &m, int &nnz, int *&row_offsets, int *&column_indices, WeightT *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading RMAT (.gr) input file %s\n", gr);
	std::ifstream cfile;
	cfile.open(gr);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == 'c') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	char sp[3];
	sscanf(str.c_str(), "%c %s %d %d", &c, sp, &m, &nnz);
	vector<vector<Edge> > vertices;
	vector<Edge> neighbors;
	for (int i = 0; i < m; i++)
		vertices.push_back(neighbors);
	int src, dst;
	for (int i = 0; i < nnz; i++) {
		getline(cfile, str);
		sscanf(str.c_str(), "%c %d %d", &c, &src, &dst);
		if (c != 'a')
			printf("line %d\n", __LINE__);
		src--;
		dst--;
		Edge e1, e2;
		if(symmetrize) {
			e2.dst = src; e2.wt = 1;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = 1;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = 1;
			vertices[dst].push_back(e1);
		}
	}
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer *.graph file to CSR format
void graph2csr(char *graph, int &m, int &nnz, int *&row_offsets, int *&column_indices, WeightT *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading .graph input file %s\n", graph);
	std::ifstream cfile;
	cfile.open(graph);
	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%d %d", &m, &nnz);
	vector<vector<Edge> > vertices;
	vector<Edge> neighbors;
	for (int i = 0; i < m; i++)
		vertices.push_back(neighbors);
	int dst;
	for (int src = 0; src < m; src ++) {
		getline(cfile, str);
		istringstream istr;
		istr.str(str);
		while(istr>>dst) {
			dst --;
			Edge e1, e2;
			if(symmetrize) {
				e2.dst = src; e2.wt = 1;
				vertices[dst].push_back(e2);
				transpose = false;
			}
			if(!transpose) {
				e1.dst = dst; e1.wt = 1;
				vertices[src].push_back(e1);
			} else {
				e1.dst = src; e1.wt = 1;
				vertices[dst].push_back(e1);
			}
		}
		istr.clear();
	}
    cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer mtx graph to CSR format
void mtx2csr(char *mtx, int &m, int &n, int &nnz, int *&row_offsets, int *&column_indices, WeightT *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading (.mtx) input file %s\n", mtx);
	std::ifstream cfile;
	cfile.open(mtx);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == '%') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	sscanf(str.c_str(), "%d %d %d", &m, &n, &nnz);
	if (m != n) {
		printf("Warning, m(%d) != n(%d)\n", m, n);
	}
	vector<vector<Edge> > vertices;
	vector<Edge> neighbors;
	for (int i = 0; i < m; i ++)
		vertices.push_back(neighbors);
	int dst, src;
	WeightT wt = 1.0f;
	for (int i = 0; i < nnz; i ++) {
		getline(cfile, str);
		int num = sscanf(str.c_str(), "%d %d %f", &src, &dst, &wt);
		if (num == 2) wt = 1;
		if (wt < 0) wt = -wt; // non-negtive weight
		src--;
		dst--;
		Edge e1, e2;
		if(symmetrize && src != dst) {
			e2.dst = src; e2.wt = wt;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = wt;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = wt;
			vertices[dst].push_back(e1);
		}
	}
	cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}
/*
void sort_neighbors(int m, int *row_offsets, int *&column_indices) {
	vector<int> neighbors;
	#pragma omp parallel for
	for(int i = 0; i < m; i++) {
		int row_begin = row_offsets[i];
		int row_end = row_offsets[i + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			neighbors.push_back(column_indices[offset]);
		}
		std::sort(neighbors.begin(), neighbors.end());
		int k = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			column_indices[offset] = neighbors[k++];
		}
	}	
}
*/
void read_graph(int argc, char *argv[], int &m, int &n, int &nnz, int *&row_offsets, int *&column_indices, int *&degree, WeightT *&weight, bool is_symmetrize=false, bool is_transpose=false, bool sorted=true, bool remove_selfloops=true, bool remove_redundents=true) {
	//if(is_symmetrize) printf("Requiring symmetric graphs for this algorithm\n");
	if (strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], m, n, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else if (strstr(argv[1], ".graph"))
		graph2csr(argv[1], m, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else if (strstr(argv[1], ".gr"))
		gr2csr(argv[1], m, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else { printf("Unrecognizable input file format\n"); exit(0); }
	printf("Calculating degree...");
	degree = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++) {
		degree[i] = row_offsets[i + 1] - row_offsets[i];
	}
	printf(" Done\n");
}

