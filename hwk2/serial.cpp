#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "grid.h"

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

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
	navg = 0;
        davg = 0.0;
	dmin = 1.0;
        //
        //  compute forces
        //
        for( int i = 0; i < n; i++ )
        {
            particles[i].ax = particles[i].ay = 0;
    //         for (int j = 0; j < n; j++ )
				// apply_force( particles[i], particles[j],&dmin,&davg,&navg);
                        // Use the grid to traverse neighbours
            int gx = grid_coord(particles[i].x);
            int gy = grid_coord(particles[i].y);

            for(int x = max(gx - 1, 0); x <= min(gx + 1, gridSize-1); x++)
            {
                for(int y = max(gy - 1, 0); y <= min(gy + 1, gridSize-1); y++)
                {
                    linkedlist_t * curr = grid.grid[x * grid.size + y];
                    while(curr != 0)
                    {
                        apply_force(particles[i], *(curr->value), &dmin, &davg, &navg);
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

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");     

    //
    // Printing summary data
    //
    if( fsum) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );    
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
