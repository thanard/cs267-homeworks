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
#define BLOCK_SIZE 128
#endif

#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset
#include <immintrin.h>

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

static void avx_mult(double* A, double* B, double* C, int lda, int ldb){
  __m256d a1 = _mm256_load_pd(A);
  __m256d a2 = _mm256_load_pd(A+lda);
  __m256d a3 = _mm256_load_pd(A+2*lda);
  __m256d a4 = _mm256_load_pd(A+3*lda);

  __m256d tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+1), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+2), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+3), tmp);
  _mm256_store_pd(C, _mm256_add_pd(_mm256_load_pd(C), tmp));

  // C+4
  tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B+ldb));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+ldb+1), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+ldb+2), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+ldb+3), tmp);
  _mm256_store_pd(C+4, _mm256_add_pd(_mm256_load_pd(C+4), tmp));    

  // C+8
  tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B+2*ldb));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+2*ldb+1), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+2*ldb+2), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+2*ldb+3), tmp);
  _mm256_store_pd(C+8, _mm256_add_pd(_mm256_load_pd(C+8), tmp));


  // C+4
  tmp = _mm256_mul_pd(a1, _mm256_broadcast_sd(B+3*ldb));
  tmp = _mm256_fmadd_pd(a2, _mm256_broadcast_sd(B+3*ldb+1), tmp);
  tmp = _mm256_fmadd_pd(a3, _mm256_broadcast_sd(B+3*ldb+2), tmp);
  tmp = _mm256_fmadd_pd(a4, _mm256_broadcast_sd(B+3*ldb+3), tmp);
  _mm256_store_pd(C+12, _mm256_add_pd(_mm256_load_pd(C+12), tmp));

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

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */ 

static void do_block (int lda, int ldb, int ldc, int M, int N, int K, double* A, double* B, double* C)
{
  double static tempA[BLOCK_SIZE * BLOCK_SIZE * sizeof(double)] __attribute__((aligned(32))); 

  int num_blk = BLOCK_SIZE/4;
  for(int k = 0; k<K; k += 4){
    for (int i =0; i<M; i += 4){
      for (int y = 0; y<4; ++y){
        for (int x = 0; x<4; ++x){
          tempA[k*4 + i*BLOCK_SIZE + x + 4*y] = A[(i+x) + (k+y)*lda];
        }
      }
    }
  }

  // print_matrix(tempA, M, K, 4);
  // print_matrix(tempA + 16, 4, 4, 4);
  // print_matrix(A, M, K, lda);

  double tempC[16];
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
        // writeto4by4(tempA, A, i, k, lda);
        // writeto4by4(tempB, B, k, j, lda);
        // // printf("i = %d, j = %d, k = %d, A = %.3lf, B = %.3lf, C = %.3lf \n", i, j, k, *(tempA+i+k*lda), *(tempB+k+j*lda), *(tempC+i+j*lda));
        // avx_mult(tempA, tempB, tempC);
        avx_mult(tempA + 4*k + i*BLOCK_SIZE, B+k+j*ldb, tempC, 4, ldb);
      }
      // Add back to C matrix
      addfrom4by4(tempC, C, i, j, ldc);
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
    // double* Cloc = NULL;
    // Cloc = (double*) malloc ((lda/4) * (lda/8) * sizeof(double));
    // memset (Cloc, 0, (lda/4) * (lda/8) * sizeof(double));
    for (int j = 0; j < lda/8; j += BLOCK_SIZE)
    /* For each block-column of B */
    for (int i = 0; i < lda/4; i += BLOCK_SIZE)
      {
        /* Accumulate block dgemms into block of C */
        int M = min (BLOCK_SIZE, lda/4-i);
        int N = min (BLOCK_SIZE, lda/8-j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int K = min (BLOCK_SIZE, lda-k);

          /* Perform individual block dgemm */
          do_block(lda, lda, lda, M, N, K, A + i + bb*lda/4 + k*lda, 
            B + k + (j+ aa*(lda/8))*lda, C + i + bb*lda/4 + (j + aa*(lda/8))*lda);
          //do_block(lda,lda,lda, M, N, K, A + i + bb*lda/4 + k*lda, B + k + (j+ aa*(lda/8))*lda, Cloc + i + j*(lda/4));
        }
        //#pragma omp critical
        // for (int jj = 0; jj<lda/8; jj++)
        //  for (int ii = 0; ii<lda/4; ii++)
        //    C[i + bb*lda/4 + ii + (j + aa*(lda/8) + jj)*lda] += Cloc[ii + jj*(lda/4)];
      }

  }
}
