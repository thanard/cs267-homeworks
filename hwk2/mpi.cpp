#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
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
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    int nlocal = 0;
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    MPI_Status status, status2;
    MPI_Request request, request2;
    set_size( n );
    double pool_size = get_size()/n_proc;
    double cutoff = get_cutoff();
    //particle_t* pool_local = (particle_t*) malloc(n * sizeof(particle_t));
    // printf("cutoff: %f, n = %d\n", cutoff, n);
    particle_t pool_local[n];
    int pool_idx, j;

    if( rank == 0 ){
        init_particles( n, particles );
        // particle_t* pool = (particle_t*) malloc(n * n_proc * sizeof(particle_t));
        particle_t pool[n*n_proc]; 
        //int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
        int partition_sizes[n_proc] = {0};
        for(int i=0; i<n; i++){
            pool_idx = int(particles[i].y/pool_size);
            //if (pool_idx >= n_proc)
            //printf("SOMETHING WRONG, Pool index: %d\n", pool_idx);

            j = partition_sizes[pool_idx];
            pool[j + n*pool_idx] = particles[i];
            if (pool_idx==0)
                pool_local[j] = particles[i];
            partition_sizes[pool_idx] += 1;
            // printf("%d %d\n", partition_sizes[pool_idx], pool_idx);
        }
        nlocal = partition_sizes[0];

        for(int i=1; i<n_proc; i++)
            MPI_Send(&pool[n*i], partition_sizes[i], PARTICLE, i, i, MPI_COMM_WORLD);
        // free( partition_sizes );
    }
    else
    {
        MPI_Recv(&pool_local[0], n, PARTICLE, 0, rank, MPI_COMM_WORLD, &status);
        // MPI_Recv(&nlocal, 1, MPI_INT, 0, rank+n_proc, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, PARTICLE, &nlocal);
    }
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    particle_t* local_lowerband = (particle_t*) malloc(n * sizeof(particle_t));
    particle_t* local_upperband = (particle_t*) malloc(n * sizeof(particle_t));
    particle_t* sending_up = (particle_t*) malloc(n * sizeof(particle_t));
    particle_t* sending_lower = (particle_t*) malloc(n * sizeof(particle_t));
    int local_n_lowerband = 0;
    int local_n_upperband = 0;

    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        // 
        //  collect all global data locally (not good idea to do)
        //
        // MPI_Allgatherv( pool_local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
        // printf("step: %d\n",step); 
        if(rank >=1){
            int n_lowerband = 0;
            particle_t* tmp_lowerband = (particle_t*) malloc(nlocal * sizeof(particle_t)); 
            for(int i=0; i<nlocal; i++){
                if (pool_size * rank + cutoff >= pool_local[i].y){
                    tmp_lowerband[n_lowerband] = pool_local[i];
                    n_lowerband += 1;
                }
            }

            // printf("rank nlocal n_lowerband %d %d %d\n", rank, nlocal, n_lowerband);
            MPI_Isend(tmp_lowerband, n_lowerband, PARTICLE, rank-1, rank-1, MPI_COMM_WORLD, &request);
            MPI_Recv(local_lowerband, n, PARTICLE, rank-1, rank, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &local_n_lowerband);
        }
        if (rank < n_proc-1){
            int n_upperband = 0;
            particle_t* tmp_upperband = (particle_t*) malloc(nlocal * sizeof(particle_t));
            for(int i=0; i<nlocal; i++){
                if (pool_size*(rank +1) - cutoff < pool_local[i].y) {
                    tmp_upperband[n_upperband] = pool_local[i];
                    n_upperband += 1;
                }
            }
            MPI_Isend(tmp_upperband, n_upperband, PARTICLE, rank+1, rank+1, MPI_COMM_WORLD, &request2);
            MPI_Recv(local_upperband, n, PARTICLE, rank+1, rank, MPI_COMM_WORLD, &status2);
            MPI_Get_count(&status2, PARTICLE, &local_n_upperband);
        }
        printf("step = %d, rank = %d, nlocal = %d, n_upper = %d, n_lower = %d\n", step, rank, nlocal, local_n_upperband, local_n_lowerband);
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        //
        //  compute all forces
        //
        for( int i = 0; i < nlocal; i++ )
        {
            pool_local[i].ax = pool_local[i].ay = 0;
            for (int j = 0; j < nlocal; j++ )
                apply_force( pool_local[i], pool_local[j], &dmin, &davg, &navg );
            for (int j = 0; j < local_n_lowerband; j++ )
                apply_force( pool_local[i], local_lowerband[j], &dmin, &davg, &navg );
            for (int j = 0; j < local_n_upperband; j++ )
                apply_force( pool_local[i], local_upperband[j], &dmin, &davg, &navg );
            if (pool_local[i].ay >1000){
                printf("TO Large a = %f\n", pool_local[i].ay);
                printf("i = %d, rank = %d, nlocal = %d, n_upper = %d, n_lower = %d\n", i, rank, nlocal, local_n_upperband, local_n_lowerband);
            }
        }
    
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        // for (int i=0; i<nlocal; i++){
        //     if (rank ==2)
        //         printf ("Before y: %f %f %f, i=%d\n", pool_local[i].y, pool_local[i].vy, pool_local[i].ay, i);}
        int k = 0;
        int sending_n_up = 0;
        int sending_n_lower = 0;
        for( int i = 0; i < nlocal; i++ ){
            move( pool_local[i]);
            // Update particles in different procs
            int flag = int(pool_local[i].y/pool_size);
            //if (rank ==3)
                // printf ("After y: %f %f\n", pool_local[i].y, pool_size);
            if (flag == rank+1){
                sending_up[sending_n_up] = pool_local[i];
                ++sending_n_up;
            }else if (flag == rank-1){
                sending_lower[sending_n_lower] = pool_local[i];
                ++sending_n_lower;
            }else if(flag != rank){
                printf("ERROR, RANK = %d, FLAG = %d, INDEX = %d, STEP = %d\n", rank, flag, i, step);
                printf ("After y: %f %f %f, pool_size=%f\n", pool_local[i].y, pool_local[i].vy, pool_local[i].ay, pool_size);
                exit(0);
            }
            else{
                pool_local[k] = pool_local[i];
                k++;
            }
            //printf("Rank n_up n_down: %d %d %d\n",rank, sending_n_up, sending_n_lower);
        }
        nlocal = k;

        if(rank>0){
            int tmp = 0;
            MPI_Isend(sending_lower, sending_n_lower, PARTICLE, rank-1, rank-1 + n_proc, MPI_COMM_WORLD, &request);
            MPI_Recv(&pool_local[nlocal], n-nlocal, PARTICLE, rank-1, rank, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &tmp);
            nlocal += tmp;
        }
        if(rank<n_proc-1){
            int tmp = 0;
            MPI_Isend(sending_up, sending_n_up, PARTICLE, rank+1, rank +1, MPI_COMM_WORLD, &request2);
            MPI_Recv(&pool_local[nlocal], n-nlocal, PARTICLE, rank+1, rank + n_proc, MPI_COMM_WORLD, &status2);
            MPI_Get_count(&status2, PARTICLE, &tmp);
            nlocal += tmp;
        }
        // printf("step: %d, rank: %d, nlocal: %d\n", step, rank, nlocal);
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
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
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    // free( partition_offsets );
    //free( pool_local );
    if (rank==0)
        free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
