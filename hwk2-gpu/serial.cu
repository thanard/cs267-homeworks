#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "common.h"


struct grid
{
    int size;
    linkedlist_t ** grid;
};

typedef struct grid grid_t;

//
// initialize grid and fill it with particles
// 
void grid_init(grid_t & grid, int size)
{
    grid.size = size;

    // Initialize grid
    grid.grid = (linkedlist**) malloc(sizeof(linkedlist*) * size * size);

    if (grid.grid == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the grid!\n");
        exit(1);
    }

    memset(grid.grid, 0, sizeof(linkedlist*) * size * size);
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
    newElement->next = grid.grid[gridCoord];

    grid.grid[gridCoord] = newElement;
    // End of critical section
}

//
// Removes a particle from a grid
//
bool grid_remove(grid_t & grid, particle_t * p, int gridCoord)
{
    if (gridCoord == -1)
        gridCoord = grid_coord_flat(grid.size, p->x, p->y);

    // No elements?
    if (grid.grid[gridCoord] == 0)
    {
        return false;
    }

    // Beginning of critical section

    linkedlist_t ** nodePointer = &(grid.grid[gridCoord]);
    linkedlist_t * current = grid.grid[gridCoord];

    while(current && (current->value != p))
    {
        nodePointer = &(current->next);
        current = current->next;
    }

    if (current)
    {
        *nodePointer = current->next;
        free(current);
    }

    // End of critical section

    return !!current;
}


//
//  benchmarking program
//
int main( int argc, char **argv )
{    
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
    set_size( n );
    init_particles( n, particles );
    
    // Set up grids
    int gridSize = (get_size()/get_cutoff()) + 1; // TODO: Rounding errors?
    grid_t grid;
    grid_init(grid, gridSize);
    for (int i = 0; i < n; ++i)
    {
        grid_add(grid, &particles[i]);
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //
        for( int i = 0; i < n; i++ )
        {
            particles[i].ax = particles[i].ay = 0;

            int gx = grid_coord(particles[i].x);
            int gy = grid_coord(particles[i].y);

            for(int x = max(gx - 1, 0); x <= min(gx + 1, gridSize-1); x++)
            {
                for(int y = max(gy - 1, 0); y <= min(gy + 1, gridSize-1); y++)
                {
                    linkedlist_t * curr = grid.grid[x * grid.size + y];
                    while(curr != 0)
                    {
                        apply_force(particles[i], *(curr->value));
                        curr = curr->next;
                    }
                }
            }
        }
 
        //
        //  move particles
        //
        for( int i = 0; i < n; i++ ) 
        {
            int gc = grid_coord_flat(grid.size, particles[i].x, particles[i].y);

            move(particles[i]);

            // Re-add the particle if it has changed grid position
            if (gc != grid_coord_flat(grid.size, particles[i].x, particles[i].y))
            {
                if (! grid_remove(grid, &particles[i], gc))
                {
                    fprintf(stdout, "Error: Failed to remove particle '%p'. Code must be faulty. Blame source writer.\n", &particles[i]);
                    exit(3);
                }
                grid_add(grid, &particles[i]);
            }
        }
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
