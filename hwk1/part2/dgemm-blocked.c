/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 512
#endif

#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      for (int k = 0; k < K; ++k)
	      C[i+j*(lda/4)] += A[i+k*lda] * B[k+j*lda]; // Changed from C[i+j*(lda/8)]
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  omp_set_num_threads(32);
  #pragma omp parallel 
  {
    int id = omp_get_thread_num();
    //printf("%d\n", id);
    int aa = id/4;
    int bb = id%4;
    // int nthrds = omp_get_num_threads();
    // printf("%d\n", nthrds);
    // #pragma omp for
    /* For each block-row of A */ 
    double* Cloc = NULL;
    Cloc = (double*) malloc ((lda/4) * (lda/8) * sizeof(double));
    memset (Cloc, 0, (lda/4) * (lda/8) * sizeof(double));
    for (int i = 0; i < lda/4; i += BLOCK_SIZE)
      /* For each block-column of B */
      for (int j = 0; j < lda/8; j += BLOCK_SIZE)
      {
        /* Accumulate block dgemms into block of C */
        int M = min (BLOCK_SIZE, lda/4-i);
        int N = min (BLOCK_SIZE, lda/8-j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
        	/* Correct block dimensions if block "goes off edge of" the matrix */
        	int K = min (BLOCK_SIZE, lda-k);

        	/* Perform individual block dgemm */
        	//do_block(lda, M, N, K, A + i + bb*lda/4 + k*lda, B + k + (j+ aa*(lda/8))*lda, C + i + bb*lda/4 + (j + aa*(lda/8))*lda);
        	do_block(lda, M, N, K, A + i + bb*lda/4 + k*lda, B + k + (j+ aa*(lda/8))*lda, Cloc + i + j*(lda/4));
        }
        //#pragma omp critical
        for (int jj = 0; jj<lda/8; jj++)
        	for (int ii = 0; ii<lda/4; ii++)
        		C[i + bb*lda/4 + ii + (j + aa*(lda/8) + jj)*lda] += Cloc[ii + jj*(lda/4)];
      }

  }
}
