// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "graph_io_block.h"

int main(int argc, char *argv[]) {
	printf("Sparse Matrix-Vector Multiplication by Xuhao Chen\n");
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 2) {
		printf("Usage: %s <graph> [is_directed(0/1)]\n", argv[0]);
		exit(1);
	} else if (argc> 2) {
		is_directed = atoi(argv[2]);
		if(is_directed) printf("A is not a symmetric matrix\n");
		else printf("A is a symmetric matrix\n");
	}
	if(!is_directed) symmetrize = true;

	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	ValueType *h_weight = NULL;
	int *block = NULL, *block_row = NULL, *row_start = NULL;
	int num_block_all;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, block, block_row, row_start, num_block_all, symmetrize, false, true, false, true);

	int height=2;
	int width=2;
	int num_rows = m;
	int num_cols = m;
	ValueType *h_x = (ValueType *)malloc(m * sizeof(ValueType));
	ValueType *h_y = (ValueType *)malloc(m * sizeof(ValueType));
	//ValueType **value;
	//value = (ValueType **)malloc(num_block_all * sizeof(float *));
	//for(int i = 0; i < num_block_all; i ++)
	//	value[i] = (ValueType *)malloc(sizeof(ValueType) * height * width);
	//ValueType *h_row_start = (ValueType *)malloc(num_block_all * sizeof(ValueType));
	ValueType *value = (ValueType *)malloc(num_block_all * 4 * sizeof(float));
	ValueType *y_host = (ValueType *)malloc(m * sizeof(ValueType));
	srand(13);
	for(int i = 0; i < nnz; i++) h_weight[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); // Ax[] (-1 ~ 1)
//	for(int i = 0; i < nnz; i++) h_weight[i] = rand() / (RAND_MAX + 1.0); // Ax[] (0 ~ 1)
	for(int i = 0; i < num_cols; i++) h_x[i] = rand() / (RAND_MAX + 1.0);
	for(int i = 0; i < num_rows; i++) {
		h_y[i] = rand() / (RAND_MAX + 1.0);
		y_host[i] = h_y[i];
	}


	//int value[num_block_all][height*width]={0};
//	float **value;
  //	vector<vector<float> > value;
		printf("value initialization_1\n");
	for(int j = 0; j < num_block_all; j ++){
		for(int i = 0; i < height*width; i ++){
			value[j * 4 + i] = 0.0;
		}
	}
		
		printf("num_block_all: %d\n", num_block_all);
		printf("value initialization_2\n");

	for(int j = 0; j < 4; j ++){
		for(int i = 0; i < height*width; i ++){
			printf("value: %f \n", value[j*4+i]);
		}
	}

		printf("value initialization_3\n");

	for(int j = 0; j < 4; j ++){
			printf("block_row: %d \n", block_row[j]);
	}

//	for(int j = 50; j < 55; j ++){
//			printf("block: %d \n", block[j]);
//	}
	
	//int one_matched = 0;
	for(int j = 0; j < num_block_all; j ++){
		//for(int i = 0; i < num_rows/height; i ++){
		int i = block_row[j];

			//one_matched = 0;
			for(int offset = h_row_offsets[i]; offset < h_row_offsets[i+height]; offset ++){		
				//if((h_column_indices[offset]/2 == block[j]) && (block_row[j] == i)){
				if(h_column_indices[offset]/2 == block[j]){
					if((h_column_indices[offset]%2 == 0) && (offset < h_row_offsets[i+1]))
						value[4 * j + 0] = h_weight[offset];
					else if((h_column_indices[offset]%2 == 0) && (offset > h_row_offsets[i+1]))
						value[4 * j + 2] = h_weight[offset];
					else if((h_column_indices[offset]%2 == 1) && (offset < h_row_offsets[i+1]))
						value[4 * j + 1] = h_weight[offset];
					else if((h_column_indices[offset]%2 == 1) && (offset > h_row_offsets[i+1]))
						value[4 * j + 3] = h_weight[offset];
						//one_matched ++;
		//printf("value initialization_3.5\n");
				}
		//printf("value initialization_3.6\n");
			}
		//printf("value initialization_3.7\n");
		//if(one_matched != 0)
		//	break;

		//}
		//printf("value initialization_3.8\n");
	}

		printf("value initialization_4\n");

/*	for(int j = num_block_all-1; j >= 0 ; j --){
		h_block[j] = *block.end();
		block.pop_back();
	}
	for(int j = num_rows / 2; j >= 0; j --){
		h_row_start[j] = *row_start.end();
		row_start.pop_back();
	}
*/
//	for(int j = num_block_all - 1; j >= 0 ; j --){
//		for(int i = height * width - 1; i >= 0 ; i --){		
//			h_block[j] = *block.end();
//			block.pop_back();
//		}
//	}


//	for(int i = 0; i < 100; i++)
//		printf("%f ", y_host[i]);

	SpmvSolver(m, nnz, h_row_offsets, h_column_indices, h_weight, h_x, h_y, value, block, row_start, num_block_all);
	SpmvVerifier(m, nnz, h_row_offsets, h_column_indices, h_weight, h_x, y_host, h_y);
	
	free(h_row_offsets);
	free(h_column_indices);
	free(block);
	free(value);
	free(block_row);
	free(row_start);
	free(h_degree);
	free(h_weight);
	free(h_x);
	free(h_y);
	return 0;
}
