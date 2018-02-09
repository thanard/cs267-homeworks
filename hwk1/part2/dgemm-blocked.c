#include <stdio.h>
#include <string.h>
#include <immintrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
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
  for (int x = 0; x<leftover_collumn; ++x){
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
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */ 

double static tempA[BLOCK_SIZE * BLOCK_SIZE * sizeof(double)] __attribute__((aligned(64)));  

static void do_block (int lda, int ldb, int ldc, int M, int N, int K, double* A, double* B, double* restrict C)
{
  for(int k = 0; k<K; k += 4){
    for (int i =0; i<M; i += 4){
      int leftover_collumn = min(4, M-i);
      int leftover_row = min(4, K-k);
      
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
      }
    }
  }

  /* For each row i of A */
  for (int j = 0; j < N; j += 4)
    /* For each column j of B */ 
    for (int i = 0; i < M; i+=4)
    {
      // Reset tempC to zero.
      double tempC[16]={0};
      for (int k = 0; k < K; k += 4)
      {
        avx_mult(tempA + 4*k + i*BLOCK_SIZE, B+k+j*ldb, tempC, 4, ldb);
      }
      int leftover_row = min (4,M-i);
      int leftover_collumn = min(4, N-j);
      addfrom4by4(tempC, C + i + ldc*j, ldc, leftover_row, leftover_collumn);
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
    for (int i = 0; i < lda/4; i += BLOCK_SIZE)
      /* For each block-column of B */
      {
        /* Accumulate block dgemms into block of C */
        int M = min (BLOCK_SIZE, lda/4-i);
        int N = min (BLOCK_SIZE, lda/8-j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int K = min (BLOCK_SIZE, lda-k);

          /* Perform individual block dgemm */
          do_block(lda,lda,lda, M, N, K, A + i + bb*lda/4 + k*lda, B + k + (j+ aa*(lda/8))*lda, 
            C + i + bb*lda/4 + (j + aa*(lda/8))*lda);
          // do_block(lda, M, N, K, A + i + bb*lda/4 + k*lda, B + k + (j+ aa*(lda/8))*lda, Cloc + i + j*(lda/4));
        }
        // //#pragma omp critical
        // for (int jj = 0; jj<lda/8; jj++)
        //  for (int ii = 0; ii<lda/4; ii++)
        //    C[i + bb*lda/4 + ii + (j + aa*(lda/8) + jj)*lda] += Cloc[ii + jj*(lda/4)];
      }

  }
}