#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

inline int mymin( int a, int b ) { return a < b ? a : b; }
inline int mymax( int a, int b ) { return a > b ? a : b; }

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

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
void apply_force( particle_t &particle, particle_t &neighbor );
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