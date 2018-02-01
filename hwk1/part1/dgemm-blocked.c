/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)
/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_SIZE_2 30
#define BLOCK_SIZE 100

#define min(a,b) (((a)<(b))?(a):(b))

static void do_block_2 (int lda, int M, int N, int K, double* A, double* B, double* C)
{  
  /* For each row i of A */
  for (int j = 0; j < N; ++j)
    /* For each column j of B */ 
    for (int i = 0; i < M; ++i) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
        /* Compute C(i,j) */
        cij += A[k+i*lda] * B[k+j*lda];
      C[i+j*lda] = cij; 
    }

}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int j = 0; j < M; j += BLOCK_SIZE_2)
    /* For each column j of B */ 
    for (int i = 0; i < N; i += BLOCK_SIZE_2) 
    {
      for (int k = 0; k < K; k += BLOCK_SIZE_2)
      {
        int M_2 = min (BLOCK_SIZE_2, M-i);
        int N_2 = min (BLOCK_SIZE_2, N-j);
        int K_2 = min (BLOCK_SIZE_2, K-k);
        do_block_2(lda, M_2, N_2, K_2, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
    }
}

static void transpose (double* A, int lda){
    int i ,j;
    double tmp;
    for (j = 0; j < lda; j++) {
        for (i = 0 ; i < lda; i++) {
            tmp = A[i+j*lda];
            A[i+j*lda] = A[j+i*lda];
            A[j+i*lda] = tmp;
        }
    }  
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  transpose(A, lda);
  for (int j = 0; j < lda; j += BLOCK_SIZE)
    /* For each block-column of B */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M = min (BLOCK_SIZE, lda-i);
      	int N = min (BLOCK_SIZE, lda-j);
      	int K = min (BLOCK_SIZE, lda-k);

      	/* Perform individual block dgemm */
      	do_block(lda, M, N, K, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
}
