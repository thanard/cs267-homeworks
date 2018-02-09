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

static void avx_mult(double* A, double* B, double* restrict C, int lda, int ldb){
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

static void addfrom4by4(double* temp, double* restrict dest, int lda, int leftover_row, int leftover_collumn){
  // Write to big from small.
  // print_matrix(dest, 4, 4, lda);
  // print_matrix(temp, 4, 4, 4);
  for (int x = 0; x<leftover_collumn; ++x){
    // _mm256_store_pd(dest + lda*x, 
    //   _mm256_add_pd(_mm256_loadu_pd(dest + lda*x),
    //     _mm256_loadu_pd(temp +4*x)
    //       )
    //   );
    if (leftover_row == 4){
      dest[lda*x] += temp[4*x];
      dest[1 + lda*x] += temp[1 + 4*x];
      dest[2 + lda*x] += temp[2 + 4*x];
      dest[3 + lda*x] += temp[3 + 4*x];  
    }else if (leftover_row == 1){
      dest[lda*x] += temp[4*x];
    }else if (leftover_row == 2){
      dest[lda*x] += temp[4*x];
      dest[1 + lda*x] += temp[1 + 4*x];
    }else{
      dest[lda*x] += temp[4*x];
      dest[1 + lda*x] += temp[1 + 4*x];
      dest[2 + lda*x] += temp[2 + 4*x];
    }
    
  }
  // print_matrix(dest, 4, 4, lda);
  // *(big+i+j*lda) += *(small);
  // *(big+i+1+j*lda) += *(small+1);
  //  *(big+i+2+j*lda) += *(small+2);
  //  *(big+i+3+j*lda) += *(small+3);

  //  *(big+i+j*lda+lda) += *(small+4);
  //  *(big+i+1+j*lda+lda) += *(small+5);
  //  *(big+i+2+j*lda+lda) += *(small+6);
  //  *(big+i+3+j*lda+lda) += *(small+7);
  
  //  *(big+i+j*lda + 2*lda) += *(small+8);
  //  *(big+i+1+j*lda + 2*lda) += *(small+9);
  //  *(big+i+2+j*lda + 2*lda) += *(small+10);
  //  *(big+i+3+j*lda + 2*lda) += *(small+11);

  //  *(big+i+j*lda + 3*lda) += *(small+12);
  //  *(big+i+1+j*lda + 3*lda) += *(small+13);
  //  *(big+i+2+j*lda + 3*lda) += *(small+14);
  //  *(big+i+3+j*lda + 3*lda) += *(small+15);
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
double static tempA[BLOCK_SIZE * BLOCK_SIZE * sizeof(double)] __attribute__((aligned(64)));  

static void do_block (int lda, int ldb, int ldc, int M, int N, int K, double* A, double* B, double* restrict C)
{
  // memset(tempA, 0, sizeof(tempA));
  // printf("tempA:zero\n");
  // print_matrix(tempA, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  // double blk_B[16*BLOCK_SIZE*BLOCK_SIZE];
  // double blk_C[BLOCK_SIZE*BLOCK_SIZE];
  for(int k = 0; k<K; k += 4){
    for (int i =0; i<M; i += 4){
      int leftover_collumn = min(4, M-i);
      int leftover_row = min(4, K-k);
      // printf("A_leftover_row=%d A_leftover_collumn=%d \n", leftover_row, leftover_collumn);
      for (int y = 0; y < 4; ++y){
        if (y>= leftover_row){
          tempA[k*4 + i*BLOCK_SIZE  + 4*y] = 0;
          tempA[k*4 + i*BLOCK_SIZE  + 1 +4*y] = 0;
          tempA[k*4 + i*BLOCK_SIZE  + 2 +4*y] = 0;
          tempA[k*4 + i*BLOCK_SIZE  + 3 +4*y] = 0;
        }else{
          if (leftover_collumn==4){
              _mm256_store_pd(tempA + k*4 + i*BLOCK_SIZE + 4*y, _mm256_loadu_pd(A + i + (k+y)*lda));
          }else if (leftover_collumn==3){
             tempA[k*4 + i*BLOCK_SIZE + 4*y] = A[(i) + (k+y)*lda];
             tempA[k*4 + i*BLOCK_SIZE + 1 + 4*y] = A[(i+1) + (k+y)*lda];
             tempA[k*4 + i*BLOCK_SIZE + 2 + 4*y] = A[(i+2) + (k+y)*lda];
             tempA[k*4 + i*BLOCK_SIZE + 3 + 4*y] = 0;
          }else if (leftover_collumn==2){
             tempA[k*4 + i*BLOCK_SIZE + 4*y] = A[(i) + (k+y)*lda];
             tempA[k*4 + i*BLOCK_SIZE + 1 + 4*y] = A[(i+1) + (k+y)*lda];
             tempA[k*4 + i*BLOCK_SIZE + 2 + 4*y] = 0;
             tempA[k*4 + i*BLOCK_SIZE + 3 + 4*y] = 0;
          }else{
             tempA[k*4 + i*BLOCK_SIZE + 4*y] = A[(i) + (k+y)*lda];
             tempA[k*4 + i*BLOCK_SIZE + 1 + 4*y] = 0;
             tempA[k*4 + i*BLOCK_SIZE + 2 + 4*y] = 0;
             tempA[k*4 + i*BLOCK_SIZE + 3 + 4*y] = 0;
          }
        } 
          // tempA[k*4 + i*BLOCK_SIZE + 4*y] = A[(i) + (k+y)*lda];
          // tempA[k*4 + i*BLOCK_SIZE + 1 + 4*y] = A[(i+1) + (k+y)*lda];
          // tempA[k*4 + i*BLOCK_SIZE + 2 + 4*y] = A[(i+2) + (k+y)*lda];
          // tempA[k*4 + i*BLOCK_SIZE + 3 + 4*y] = A[(i+3) + (k+y)*lda];
      }
    }
  }
  // printf("tempA\n");
  // print_matrix(tempA, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  // print_matrix(tempA + 16, 4, 4, 4);
  // print_matrix(A, M, K, lda);

  /* For each row i of A */
  for (int j = 0; j < N; j += 4)
    /* For each column j of B */ 
    for (int i = 0; i < M; i+=4)
    {
      // Reset tempC to zero.
      double tempC[16]={0};
      // memset(tempC, 0, sizeof(tempC));
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
      // print_matrix(tempA, 4, 4, 4);
      // print_matrix(B, 4, 4, 4);
      // printf("inner: i=%d j=%d\n", i, j);
      // Add back to C matrix
      int leftover_row = min (4,M-i);
      int leftover_collumn = min(4, N-j);
      // printf("leftover_row=%d leftover_collumn=%d",leftover_row, leftover_collumn);
      // printf("tempC\n");
      // print_matrix(tempC, 4, 4, 4);
      addfrom4by4(tempC, C + i + ldc*j, ldc, leftover_row, leftover_collumn);
      // printf("C\n");
      // print_matrix(C, 5, 5, ldc);

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
    // avx_mult(A, B, C, lda);
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
  // printf("i=%d j=%d k=%d\n", i, j, k);
  /* Perform individual block dgemm */
  // printf("%d %d %d", M, N, K);
  do_block(lda, lda, lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
 //  print_matrix(C, lda, lda, lda);
 // exit(0);
}

