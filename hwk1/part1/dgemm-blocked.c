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
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE_2 30
#define BLOCK_SIZE 94
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void print_matrix(double* A, int M, int N, int lda){
  for(int i=0; i<M; ++i){
    for(int j=0; j<N; ++j){
      printf("%lf\t", *(A+i+j*lda));
    }
    printf("\n");
  }
  printf("\n");
}

static void avx_mult(double* A, double* B, double* C){
  __m256d a1 = _mm256_loadu_pd(A);
  __m256d a2 = _mm256_loadu_pd(A+4);
  __m256d a3 = _mm256_loadu_pd(A+8);
  __m256d a4 = _mm256_loadu_pd(A+12);

  __m256d tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+1), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+2), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+3), tmp);
  _mm256_store_pd(C, _mm256_add_pd(_mm256_loadu_pd(C), tmp));

  // C+4
  tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B+4));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+5), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+6), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+7), tmp);
  _mm256_store_pd(C+4, _mm256_add_pd(_mm256_loadu_pd(C+4), tmp));    

  // C+8
  tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B+8));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+9), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+10), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+11), tmp);
  _mm256_store_pd(C+8, _mm256_add_pd(_mm256_loadu_pd(C+8), tmp));

  // C+4
  tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B+12));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+13), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+14), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+15), tmp);
  _mm256_store_pd(C+12, _mm256_add_pd(_mm256_loadu_pd(C+12), tmp));

}

static void do_block_2 (int lda, int M, int N, int K, double* A, double* B, double* C)
{  
  /* For each row i of A */
  for (int j = 0; j < N; j += 4)
    /* For each column j of B */ 
    for (int i = 0; i < M; i += 4) 
    {
      // /* Compute C(i,j) */
      // double cij = C[i+j*lda];
      for (int k = 0; k < K; k+=4)
        avx_mult(A + i + k*lda, B + k + j*lda, C + i + j*lda);
      //   /* Compute C(i,j) */
      //   cij += A[i+k*lda] * B[k+j*lda];
      // C[i+j*lda] = cij; 
    }

}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i += BLOCK_SIZE_2)
    /* For each column j of B */ 
    for (int j = 0; j < N; j += BLOCK_SIZE_2) 
    {
      for (int k = 0; k < K; k += BLOCK_SIZE_2)
      {
        int M_2 = min (BLOCK_SIZE_2, M-i);
        int N_2 = min (BLOCK_SIZE_2, N-j);
        int K_2 = min (BLOCK_SIZE_2, K-k);
        do_block_2(lda, M_2, N_2, K_2, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  // print_matrix(A, lda, lda, lda);
  // print_matrix(B, lda, lda, lda);
    // avx_mult(A, B, C);
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	 // Correct block dimensions if block "goes off edge of" the matrix 
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
  // print_matrix(C, lda, lda, lda);
  // exit(0);
}