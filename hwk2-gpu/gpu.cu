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
    int size;
    linkedlist_t ** grid[9];
    // 0 3 6
    // 1 4 7
    // 2 5 8
};

typedef struct grid grid_t;

//
// initialize grid and fill it with particles
// 
void grid_init(grid_t & grid, grid_t & d_grid, int size)
{
    grid.size = size;

    // Initialize grid
    for(int i=0; i<9;i++){
      grid.grid[i] = (linkedlist**) malloc(sizeof(linkedlist*) * size * size);
      memset(grid.grid[i], 0, sizeof(linkedlist*) * size * size);
    }
    // if (grid.grid == NULL)
    // {
    //     fprintf(stderr, "Error: Could not allocate memory for the grid!\n");
    //     exit(1);
    // }

    // Set gpu grid
    for (int i=0; i<9;i++){
      cudaMalloc((void **) &d_grid.grid[i], size * size * sizeof(linkedlist*));
    }
}

//
// adds a particle pointer to the grid
//
void grid_add(grid_t & grid, particle_t * p)
{
    int gridCoord = grid_coord_flat(grid.size, p->x, p->y);

    linkedlist_t * newElement = (linkedlist_t *) malloc(sizeof(linkedlist));
    newElement->value = p;

    // Beginning of critical section
    newElement->next = grid.grid[4][gridCoord];

    grid.grid[4][gridCoord] = newElement;
    // End of critical section
}

void grid_copy_to_cuda(grid_t & grid, grid_t & d_grid){
    for (int i=0; i<9;i++){
      cudaMemcpy(d_grid.grid[i], grid.grid[i], size * size * sizeof(linkedlist*), cudaMemcpyHostToDevice);
    }
}
// //
// // Removes a particle from a grid
// //
// bool grid_remove(grid_t & grid, particle_t * p, int gridCoord)
// {
//     if (gridCoord == -1)
//         gridCoord = grid_coord_flat(grid.size, p->x, p->y);

//     // No elements?
//     if (grid.grid[gridCoord] == 0)
//     {
//         return false;
//     }

//     // Beginning of critical section

//     linkedlist_t ** nodePointer = &(grid.grid[gridCoord]);
//     linkedlist_t * current = grid.grid[gridCoord];

//     while(current && (current->value != p))
//     {
//         nodePointer = &(current->next);
//         current = current->next;
//     }

//     if (current)
//     {
//         *nodePointer = current->next;
//         free(current);
//     }

//     // End of critical section

//     return !!current;
// }

//
//  benchmarking program
//
//
// adds a particle pointer to the grid
//
__device__ void grid_add_remove_temp_gpu(grid_t & grid, int g_sub_idx, int gridCoord, particle_t * p)
{
    // Remove
    linkedlist_t ** nodePointer = &(grid.grid[4][gridCoord]);
    linkedlist_t * current = grid.grid[4][gridCoord];

    while(current && (current->value != p))
    {
        nodePointer = &(current->next);
        current = current->next;
    }

    if (current)
    {
        *nodePointer = current->next;
    }

    // Add current to temp bin
    current->value = p;
    newElement->next = grid.grid[g_sub_idx][gridCoord];

    grid.grid[g_sub_idx][gridCoord] = current;
}

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

__global__ void compute_forces_gpu(grid_t d_grid, int n)
{
  // Get thread (particle) ID
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;
  // printf("threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", threadIdx.x,blockIdx.x, blockDim.x);
  // printf("threadIdx.y=%d, blockIdx.y=%d, blockDim.y=%d\n", threadIdx.y,blockIdx.y, blockDim.y);

  if(gx >= d_grid.size || gy >= d_grid.size) return;

  // particles[tid].ax = particles[tid].ay = 0;
  // for(int j = 0 ; j < n ; j++)
  //   apply_force_gpu(particles[tid], particles[j]);

  // get x y coord of grid from tid.
  // then look at neighbour cells.
  // do apply force.
  linkedlist_t * particle = d_grid.grid[4][gx*d_grid.size + gy];
  while( particle ){
    for(int x = max(gx - 1, 0); x <= min(gx + 1, d_grid.size-1); x++){
      for(int y = max(gy - 1, 0); y <= min(gy + 1, d_grid.size-1); y++){
        linkedlist_t * curr = d_grid.grid[4][x * d_grid.size + y];
        while(curr)
          {
              apply_force_gpu(*(particle->value), *(curr->value));
              curr = curr->next;
          }
      }
    }
    particle = particle -> next;
  }
}

__device__ double get_cutoff_gpu(){
  // double cutoff = 0.01; //TODO maybe remove this.
  return 0.01;
}

__device__ int grid_coord_gpu(double c){
  return (int)floor(c/get_cutoff_gpu());
}

__device__ int grid_coord_flat_gpu(int size, double x, double y){
      return grid_coord_gpu(x) * size + grid_coord_gpu(y);
}

__global__ void move_gpu (grid_t d_grid, int n, double frame_size)
{
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;
  int gc = gx*d_grid.size + gy;
  if(gx >= d_grid.size || gy >= d_grid.size) return;

  linkedlist_t* particle = d_grid.grid[4][gx*d_grid.size + gy];
  while( particle){

    particle_t * p = (particle->value);
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
    while( p->x < 0 || p->x > frame_size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*frame_size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > frame_size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*frame_size-p->y;
        p->vy = -(p->vy);
    }

    //
    // Check if going outside the grid.
    // Remove from current cell.
    //
    int gx_after = grid_coord_gpu(p->x);
    int gy_after = grid_coord_gpu(p->y);
    if (gx_after != gx || gy_after != gy){
      grid_add_remove_temp_gpu(d_grid, (gx_after-gx+1)*3 + gy_after-gy+1, gc, p);
    }

    particle = particle -> next;
  }

}

__global__ void add_to_grid_gpu(){

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
    grid_t grid;
    grid_t d_grid;
    grid_init(grid, d_grid, gridSize);

    for (int i = 0; i < n; ++i)
    {
        grid_add(grid, &particles[i]);
    }
    
    grid_copy_to_cuda(grid, d_grid);
    
    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //

	      // int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
        // int blks = (gridSize * gridSize + NUM_THREADS - 1) / NUM_THREADS;
        dim3 dimBlock((gridSize+15)/16, (gridSize+15)/16, 1);
        dim3 dimThread(16, 16, 1);
        // printf("blk=%d, num_threads=%d\n", blks, NUM_THREADS);
	      compute_forces_gpu <<< dimBlock, dimThread >>> (d_grid, n);

        //
        //  move particles
        //
	      move_gpu <<< dimBlock, dimThread >>> (d_grid, n, size);
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	        // Copy the particles back to the CPU
          cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
          save( fsave, n, particles);
	       }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
