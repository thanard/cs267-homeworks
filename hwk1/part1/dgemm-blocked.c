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
#include <stdio.h>
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE_2 30
#define BLOCK_SIZE 100
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void do_block_2 (int lda, int M, int N, int K, double* A, double* B, double* C)
{  
  /* For each row i of A */
  for (int k = 0; k < K; ++k)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      for (int i = 0; i < M; ++i)
      /* Compute C(i,j) */
      C[i+j*lda] += A[i+k*lda] * B[k+j*lda];
    }

}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int k = 0; k < K; k += BLOCK_SIZE_2)
    /* For each column j of B */ 
    for (int j = 0; j < N; j += BLOCK_SIZE_2) 
    {
      for (int i = 0; i < M; i += BLOCK_SIZE_2)
      {
        int M_2 = min (BLOCK_SIZE_2, M-i);
        int N_2 = min (BLOCK_SIZE_2, N-j);
        int K_2 = min (BLOCK_SIZE_2, K-k);
        do_block_2(lda, M_2, N_2, K_2, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
}

// static void avx_mult(double* A, double*B){
//   for (int i=0; i<4; i++)
//   {
//     __m256d a1 = _mm256_loadu_pd(A);
//     __m256d a2 = _mm256_loadu_pd(A+4);
//     __m256d a3 = _mm256_loadu_pd(A+8);
//     __m256d a4 = _mm256_loadu_pd(A+12);

//     __m256d b1 = _mm256_loadu_pd(B);
//     __m256d b2 = _mm256_loadu_pd(B+4);
//     __m256d b3 = _mm256_loadu_pd(B+8);
//     __m256d b4 = _mm256_loadu_pd(B+12);
//   }



// }

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}