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
#include <string.h>
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE_2 32
#define BLOCK_SIZE 128
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void print_matrix(double* A, int M, int N, int lda){
  for(int i=0; i<M; ++i){
    for(int j=0; j<N; ++j){
      printf("%.3lf\t", *(A+i+j*lda));
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

static void writeto4by4(double* small, double* big, int i, int j, int lda){
  // Write to small from big.
  *(small) = *(big+i+j*lda);
  *(small+1) = *(big+i+1+j*lda);
  *(small+2) = *(big+i+2+j*lda);
  *(small+3) = *(big+i+3+j*lda);

  *(small+4) = *(big+i+j*lda+lda);
  *(small+5) = *(big+i+1+j*lda+lda);
  *(small+6) = *(big+i+2+j*lda+lda);
  *(small+7) = *(big+i+3+j*lda+lda);
  
  *(small+8) = *(big+i+j*lda + 2*lda);
  *(small+9) = *(big+i+1+j*lda + 2*lda);
  *(small+10) = *(big+i+2+j*lda + 2*lda);
  *(small+11) = *(big+i+3+j*lda + 2*lda);

  *(small+12) = *(big+i+j*lda + 3*lda);
  *(small+13) = *(big+i+1+j*lda + 3*lda);
  *(small+14) = *(big+i+2+j*lda + 3*lda);
  *(small+15) = *(big+i+3+j*lda + 3*lda);
}

static void addfrom4by4(double* small, double* big, int i, int j, int lda){
  // Write to big from small.
  *(big+i+j*lda) += *(small);
  *(big+i+1+j*lda) += *(small+1);
   *(big+i+2+j*lda) += *(small+2);
   *(big+i+3+j*lda) += *(small+3);

   *(big+i+j*lda+lda) += *(small+4);
   *(big+i+1+j*lda+lda) += *(small+5);
   *(big+i+2+j*lda+lda) += *(small+6);
   *(big+i+3+j*lda+lda) += *(small+7);
  
   *(big+i+j*lda + 2*lda) += *(small+8);
   *(big+i+1+j*lda + 2*lda) += *(small+9);
   *(big+i+2+j*lda + 2*lda) += *(small+10);
   *(big+i+3+j*lda + 2*lda) += *(small+11);

   *(big+i+j*lda + 3*lda) += *(small+12);
   *(big+i+1+j*lda + 3*lda) += *(small+13);
   *(big+i+2+j*lda + 3*lda) += *(small+14);
   *(big+i+3+j*lda + 3*lda) += *(small+15);
}

static void do_block_2 (int lda, int M, int N, int K, double* A, double* B, double* C)
{  
  double tempA[16], tempB[16], tempC[16];
  /* For each row i of A */
  for (int j = 0; j < N; j += 4)
    /* For each column j of B */ 
    for (int i = 0; i < M; i+=4)
    {
      // Reset tempC to zero.
      memset(tempC, 0, sizeof(tempC));
      // print_matrix(tempC, 4, 4, 1);
      /* Compute C(i,j) */
      for (int k = 0; k < K; k += 4)
      {
        writeto4by4(tempA, A, i, k, lda);
        writeto4by4(tempB, B, k, j, lda);
        // printf("i = %d, j = %d, k = %d, A = %.3lf, B = %.3lf, C = %.3lf \n", i, j, k, *(tempA+i+k*lda), *(tempB+k+j*lda), *(tempC+i+j*lda));
        avx_mult(tempA, tempB, tempC);
      }
      // Add back to C matrix
      addfrom4by4(tempC, C, i, j, lda);
    }

}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int j = 0; j < N; j += BLOCK_SIZE_2)
    /* For each column j of B */ 
    for (int i = 0; i < M; i += BLOCK_SIZE_2)
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
  for (int j = 0; j < lda; j += BLOCK_SIZE)
    /* For each block-column of B */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
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
//   print_matrix(C, lda, lda, lda);
  // exit(0);
}