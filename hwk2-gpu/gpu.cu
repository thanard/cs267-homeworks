#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

extern double size;


struct grid
{
    int *size;
    linkedlist_t ** grid;
};

typedef struct grid grid_t;

__device__ double get_cutoff_gpu(){
  // double cutoff = 0.01; //TODO maybe remove this.
  return 0.01;
}

__device__ int grid_coord_gpu(double c){
  return (int)floor(c/get_cutoff_gpu());
}

__device__ int grid_coord_flat_gpu(int gridsize, double x, double y){
      return grid_coord_gpu(x) * gridsize + grid_coord_gpu(y);
}

//
//  benchmarking program
//

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, grid_t d_grid, int n, int gridsize)
{
  // Get thread (particle) ID
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;
  // printf("threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", threadIdx.x,blockIdx.x, blockDim.x);
  // printf("threadIdx.y=%d, blockIdx.y=%d, blockDim.y=%d\n", threadIdx.y,blockIdx.y, blockDim.y);
  // printf("gx=%d, gy=%d\n", gx, gy);
  // printf("grid_size=%d\n", *(d_grid.size));
  if(gx >= gridsize || gy >= gridsize) return;

  // particles[tid].ax = particles[tid].ay = 0;
  // for(int j = 0 ; j < n ; j++)
  //   apply_force_gpu(particles[tid], particles[j]);

  // get x y coord of grid from tid.
  // then look at neighbour cells.
  // do apply force.
  // printf("test1\n");
  linkedlist_t * particle = d_grid.grid[gx*gridsize + gy];
  while( particle ){
    for(int x = max(gx - 1, 0); x <= min(gx + 1, (gridsize)-1); x++){
      for(int y = max(gy - 1, 0); y <= min(gy + 1, (gridsize)-1); y++){
        linkedlist_t * curr = d_grid.grid[x * (gridsize) + y];
        // printf("d_grid.grid=%p\n", particle);
        // printf("neigh_curr=%p\n", curr);
        while(curr)
          {
              // printf("curr=%p\n", curr);
              apply_force_gpu(*(particle->value), *(curr->value));
              curr = curr->next;
          }
      }
    }
    particle = particle -> next;
  }
  // printf("end\n");
}

__global__ void move_gpu (particle_t * particles, int n, double size)//size is frame size.
{
  // printf("Test\n");

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
  //
  //  slightly simplified Velocity Verlet integration
  //  conserves energy better than explicit Euler method
  //
  p->vx += p->ax * dt;
  p->vy += p->ay * dt;
  p->x  += p->vx * dt;
  p->y  += p->vy * dt;

  //
  //  bounce from walls
  //
  while( p->x < 0 || p->x > size )
  {
      p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
      p->vx = -(p->vx);
  }
  while( p->y < 0 || p->y > size )
  {
      p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
      p->vy = -(p->vy);
  }

}


__global__ void update_grid_gpu(particle_t * particles, grid_t d_grid, grid_t d_grid2, int n, int gridSize){
  // printf("%d\n", gridSize);
  // printf("%d %d %d\n", threadIdx.x, blockIdx.x, blockDim.x);
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;
  // printf("gx=%d, gy=%d, gridsize=%d\n", gx, gy, gridSize);
  if(gx >= gridSize || gy >= gridSize) return;
  // Add all elements in grid.
  linkedlist_t *q = d_grid2.grid[gx*gridSize + gy];
  for(int i=0; i<n;i++){
    particle_t* p = &particles[i];
    printf("particle x, y=%f %f, grid_size= %d\n", p->x, p -> y, gridSize);

    if (grid_coord_flat_gpu(gridSize, p-> x, p->y) == gridSize*gx + gy){
      q->value = p;
      printf("particle x, y=%f %f\n", p->x, p -> y);
      q = q->next;
    }
  }
}

__global__ void copy_d_grids_gpu(grid_t d_grid, grid_t d_grid2, int gridSize){
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  if(gx >= gridSize || gy >= gridSize) return;
  // Add all elements in grid.
  linkedlist_t *p = d_grid2.grid[gx*gridSize + gy];
  linkedlist_t *q = d_grid.grid[gx*gridSize + gy];
  while (p){
    q->value = p->value;
    p = p->next;
    q = q->next;
  }
}

__global__ void print_gpu(int * x){
  printf("%d\n" , *x);
}

int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    // Set up grids
    int gridSize = (get_size()/get_cutoff()) + 1;
    grid_t d_grid, d_grid2;
    dim3 dimBlock((gridSize+15)/16, (gridSize+15)/16, 1);
    dim3 dimThread(16, 16, 1);

    cudaMalloc((void **)&d_grid.size, sizeof(int));
    cudaMalloc((void **) &d_grid.grid, gridSize * gridSize * sizeof(linkedlist*));
    cudaMalloc((void **) &d_grid2.grid, gridSize * gridSize * sizeof(linkedlist*));

    // Add particles to CPU grids
    // for (int i = 0; i < n; ++i)
    // {
    //     grid_add(grid, &particles[i]);
    // }

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Initialize cuda grid
    // cudaMemcpy(d_grid.size, &gridSize, sizeof(int), cudaMemcpyHostToDevice);
    printf("%d \n", gridSize);
    printf("%d \n", n);
    // print_gpu<<< 1, 1 >>> (d_grid.size);

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    update_grid_gpu <<< dimBlock, dimThread >>> (d_particles, d_grid, d_grid2, n, gridSize);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    // for( int step = 0; step < NSTEPS; step++ )
    // {
    //     //
    //     //  compute forces
    //     //
	   //    compute_forces_gpu <<< dimBlock, dimThread >>> (d_particles, d_grid, n, gridSize);

    //     //
    //     //  move particles
    //     //
    //     int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    //     // printf("blks=%d\n", blks);
	   //    move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
    //     //
    //     //  Update bins
    //     //
    //     cudaFree(&d_grid2);
    //     cudaMalloc((void **) &d_grid2.grid, gridSize * gridSize * sizeof(linkedlist*));
    //     update_grid_gpu <<< dimBlock, dimThread >>> (d_particles, d_grid, d_grid2, n, gridSize);
    //     cudaFree(&d_grid);
    //     cudaMalloc((void **) &d_grid.grid, gridSize * gridSize * sizeof(linkedlist*));

    //     // printf("test1\n");

    //     copy_d_grids_gpu<<< dimBlock, dimThread >>> (d_grid, d_grid2, gridSize);
    //     // printf("test2\n");


    //     //
    //     //  save if necessary
    //     //
    //     if( fsave && (step%SAVEFREQ) == 0 ) {
	   //      // Copy the particles back to the CPU
    //       cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
    //       save( fsave, n, particles);
	   //     }
    // }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    cudaFree(&d_grid);
    cudaFree(&d_grid2);
    //free(grid);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
