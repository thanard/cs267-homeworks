#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

inline int min( int a, int b ) { return a < b ? a : b; }
inline int max( int a, int b ) { return a > b ? a : b; }

//
//  saving parameters
//
const int NSTEPS = 1000;
const int SAVEFREQ = 10;

//
// particle data structure
//
typedef struct 
{
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

//
//  timing routines
//
double read_timer( );

double get_size();
double get_cutoff();

//
//  simulation routines
//
void set_size( int n );
void init_particles( int n, particle_t *p );
void apply_force( particle_t &particle, particle_t &neighbor , double *dmin, double *davg, int *navg);
void move( particle_t &p );


//
//  I/O routines
//
FILE *open_save( char *filename, int n );
void save( FILE *f, int n, particle_t *p );

//
//  argument processing routines
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );

//
// Grid
//
struct linkedlist
{
	linkedlist * next;
	particle_t * value;
};

typedef struct linkedlist linkedlist_t;

struct grid
{
	int size;
	linkedlist_t ** grid;
};

typedef struct grid grid_t;

//
// grid routines
//

void grid_init(grid_t & grid, int gridsize);
void grid_add(grid_t & grid, particle_t * particle);
bool grid_remove(grid_t & grid, particle_t * p, int gridCoord = -1);
void grid_clear(grid_t & grid);
int  grid_size(grid_t & grid);


//
// Calculate the grid coordinate from a real coordinate
//
inline static int grid_coord(double c)
{
    return (int)floor(c / get_cutoff());
}
inline static int grid_coord_flat(int size, double x, double y)
{
    return grid_coord(x) * size + grid_coord(y);
}

#endif