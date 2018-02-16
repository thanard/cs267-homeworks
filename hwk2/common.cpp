#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#include "common.h"

double size;

//
//  tuned constants
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
//  timer
//
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//
//  keep density constant
//
void set_size( int n )
{
    size = sqrt( density * n );
}

double get_cutoff(){
    return cutoff;
}

double get_size()
{
    return size;
}

//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p )
{
    srand48( time( NULL ) );
        
    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;
    
    int *shuffle = (int*)malloc( n * sizeof(int) );
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;
    
    for( int i = 0; i < n; i++ ) 
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];
        
        //
        //  distribute particles evenly to ensure proper spacing
        //
        p[i].x = size*(1.+(k%sx))/(1+sx);
        p[i].y = size*(1.+(k/sx))/(1+sy);

        //
        //  assign random velocities within a bound
        //
        p[i].vx = drand48()*2-1;
        p[i].vy = drand48()*2-1;
    }
    free( shuffle );
}

//
//  interact two particles
//
void apply_force( particle_t &particle, particle_t &neighbor , double *dmin, double *davg, int *navg)
{

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
	if (r2 != 0)
        {
	   if (r2/(cutoff*cutoff) < *dmin * (*dmin))
	      *dmin = sqrt(r2)/cutoff;
           (*davg) += sqrt(r2)/cutoff;
           (*navg) ++;
        }
		
    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );
 
    
	
    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

//
//  integrate the ODE
//
void move( particle_t &p )
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //
    //  bounce from walls
    //
    while( p.x < 0 || p.x > size )
    {
        p.x  = p.x < 0 ? -p.x : 2*size-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size )
    {
        p.y  = p.y < 0 ? -p.y : 2*size-p.y;
        p.vy = -p.vy;
    }
}

//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

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
// clears a grid from values and deallocates any memory from the heap.
//
void grid_clear(grid_t & grid)
{
    for (int i = 0; i < grid.size*grid.size; ++i)
    {
        linkedlist_t * curr = grid.grid[i];
        while(curr != 0)
        {
            linkedlist_t * tmp = curr->next;
            free(curr);
            curr = tmp;
        }
    }
    free(grid.grid);
}

int grid_size(grid_t & grid)
{
    int count = 0;
    for (int i = 0; i < grid.size * grid.size; ++i)
    {
        linkedlist_t * curr = grid.grid[i];
        while(curr != 0)
        {
            count++;
            curr = curr->next;
        }
    }
    return count;
}