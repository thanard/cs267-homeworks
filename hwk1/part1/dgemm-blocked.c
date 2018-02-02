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
#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_SIZE_2 32
#define BLOCK_SIZE 100

#define min(a,b) (((a)<(b))?(a):(b))

void print_mat(double* A, int M, int N){
    for (int i =0; i<M; i++){
    for(int j=0; j<N; j++){
      printf("%lf\t", A[i + N*j]);
    }
    printf("\n");
  }
}

static void do_block_2 (int lda, int M, int N, int K, double* A, double* B, double* C)
{  
  // printf("M %d, N %d, K %d \n", M, N, K);
  // print_mat(A, M, K);
  // print_mat(B, N, K);
  // print_mat(C, M, N);
  /* For each row i of A */
  for (int j = 0; j < N; ++j)
    /* For each column j of B */ 
    for (int i = 0; i < M; ++i) 
    {
      /* Compute C(i,j) */
      double cij = 0;
      for (int k = 0; k < K; ++k)
        /* Compute C(i,j) */
        cij += A[k+i*lda] * B[k+j*lda];
      C[i+j*lda] = cij; 
    }
  print_mat(C, M, N);

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
        do_block_2(lda, M_2, N_2, K_2, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
    }
}

void transpose (double* A, int lda){
  int i ,j;
  double tmp;
  for (i = 0; i < lda; i++) {
      for (j = 0 ; j < i; j++) {
          tmp = A[i+j*lda];
          A[i+j*lda] = A[j+i*lda];
          A[j+i*lda] = tmp;
      }
  }    
}
double* naive_mult (int n, double* A, double* B)
{
  double* C = (int *)malloc(n*n*sizeof(double));
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
      /* Compute C(i,j) */
      double cij = 0;
      for( int k = 0; k < n; k++ )
        cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
    }
  print_mat(C, n, n);
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  print_mat(A, lda, lda);
  print_mat(B, lda, lda);
  // print_mat(C, lda, lda);
  naive_mult(lda, A, B);
  transpose(A, lda);
  /* For each block-row of A */ 
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

  // exit(0);
}
