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

//
// initialize grid and fill it with particles
// 
void grid_init(grid_t & grid, grid_t & d_grid, int size)
{
    grid.size =(int*) malloc(sizeof(int));
    *grid.size = size;
    // Initialize grid
    grid.grid = (linkedlist**) malloc(sizeof(linkedlist*) * size * size);
    memset(grid.grid, 0, sizeof(linkedlist*) * size * size);

    if (grid.grid == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the grid!\n");
        exit(1);
    }

    // Set gpu grid
    cudaMalloc(&d_grid.size, sizeof(int));
    cudaMalloc((void **) &d_grid.grid, size * size * sizeof(linkedlist*));
}

//
// adds a particle pointer to the grid
//
void grid_add(grid_t & grid, particle_t * p)
{
    // printf("%p\n", p);
    int gridCoord = grid_coord_flat(*grid.size, p->x, p->y);

    linkedlist_t * newElement = (linkedlist_t *) malloc(sizeof(linkedlist));
    newElement->value = p;

    // Beginning of critical section
    newElement->next = grid.grid[gridCoord];

    grid.grid[gridCoord] = newElement;
    // End of critical section
}

void grid_copy_to_cuda(grid_t & grid, grid_t & d_grid){
    cudaMemcpy(&d_grid.size, &grid.size, sizeof(int), cudaMemcpyHostToDevice);
    printf("%d \n", *grid.size);
    cudaMemcpy(&d_grid.grid, &grid.grid, size * size * sizeof(linkedlist*), cudaMemcpyHostToDevice);
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

__global__ void compute_forces_gpu(particle_t * particles, grid_t & d_grid, int n, int gridsize)
{
  // Get thread (particle) ID
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;
  // printf("threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", threadIdx.x,blockIdx.x, blockDim.x);
  // printf("threadIdx.y=%d, blockIdx.y=%d, blockDim.y=%d\n", threadIdx.y,blockIdx.y, blockDim.y);
  printf("gx=%d, gy=%d, grid_size=%d\n", gx, gy, *(d_grid.size));
  if(gx >= gridsize || gy >= gridsize) return;

  // particles[tid].ax = particles[tid].ay = 0;
  // for(int j = 0 ; j < n ; j++)
  //   apply_force_gpu(particles[tid], particles[j]);

  // get x y coord of grid from tid.
  // then look at neighbour cells.
  // do apply force.
  printf("test1\n");
  linkedlist_t * particle = d_grid.grid[gx*gridsize + gy];
  printf("test2\n");
  while( particle ){
    for(int x = max(gx - 1, 0); x <= min(gx + 1, (gridsize)-1); x++){
      for(int y = max(gy - 1, 0); y <= min(gy + 1, (gridsize)-1); y++){
        linkedlist_t * curr = d_grid.grid[x * (gridsize) + y];
        while(curr)
          {
              apply_force_gpu(*(particle->value), *(curr->value));
              curr = curr->next;
          }
      }
    }
    particle = particle -> next;
  }
  printf("end\n");
}

__global__ void move_gpu (particle_t * particles, int n, double size)//size is frame size.
{
    printf("Test\n");

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

// __global__ void update_grid_temp_gpu(grid_t & d_grid, grid_t & d_grid_temp, int n, double framesize){
//   // Get thread (particle) ID
//   int gx = threadIdx.x + blockIdx.x * blockDim.x;
//   int gy = threadIdx.y + blockIdx.y * blockDim.y;
//   printf("Test\n");

//   if(gx >= *d_grid.size || gy >= *d_grid.size) return;

//   linkedlist_t* new_p_node = d_grid_temp.grid[gx*d_grid_temp.size + gy];

//   for (int x = max(gx-1, 0); x <= min(gx+1, *d_grid.size-1); x++){
//     for (int y = max(gy-1, 0); y<= min(gy+1, *d_grid.size-1); y++){
//       linkedlist_t* p_node = d_grid.grid[x*(*d_grid.size) + y];
//       while(p_node){
//         particle_t * p = (p_node -> value);
//         // Add particle p to head of d_grid_temp
//         if (grid_coord_flat_gpu(*d_grid.size, p-> x, p->y) == grid_coord_flat_gpu(*d_grid.size, gx, gy)){
//           linkedlist_t * temp = new linkedlist_t;
//           temp -> value = p;
//           temp -> next = d_grid_temp.grid[gx*(*d_grid.size) + gy];
//           d_grid_temp.grid[gx*(*d_grid.size) + gy] = temp;
//         }

//         p_node = p_node -> next;
//       }      
//     }
//   }

// }

__global__ void update_grid_gpu(particle_t * particles, grid_t & d_grid, int n, double framesize){
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  if(gx >= (*d_grid.size) || gy >= *d_grid.size) return;
  // Delete all elements in grid.
  linkedlist_t * p_node = d_grid.grid[gx*(*d_grid.size) + gy];
  while(p_node){
    linkedlist_t * temp = p_node -> next;
    delete[] p_node;
    p_node = temp;
  }
  printf("Test\n");

  // Add all elements in grid.
  for(int i=0; i<n;i++){
    particle_t* p = &particles[i];
    if (grid_coord_flat_gpu(*d_grid.size, p-> x, p->y) == grid_coord_flat_gpu(*d_grid.size, gx, gy)){
      linkedlist_t * temp = new linkedlist_t;
      temp -> value = p;
      temp -> next = d_grid.grid[gx*(*d_grid.size) + gy];
      d_grid.grid[gx*(*d_grid.size) + gy] = temp;
    }

  }
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

        // int blks = (gridSize * gridSize + NUM_THREADS - 1) / NUM_THREADS;
        dim3 dimBlock((gridSize+15)/16, (gridSize+15)/16, 1);
        dim3 dimThread(16, 16, 1);
        // printf("blk=%d, num_threads=%d\n", blks, NUM_THREADS);
	      compute_forces_gpu <<< dimBlock, dimThread >>> (d_particles, d_grid, n, gridSize);

        //
        //  move particles
        //
        int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
	      move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        
        //
        //  Update bins
        //
        update_grid_gpu <<< dimBlock, dimThread >>> (d_particles, d_grid, n, size);
        // update_grid_temp_gpu <<< dimBlock, dimThread >>> (d_grid, d_grid_temp, n, size);

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
    //cudaFree(d_grid);
    //free(grid);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
