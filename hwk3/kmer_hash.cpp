#include <cstdio>
#include <cstdlib>
#include <vector>
#include <list>
#include <set>
#include <numeric>
#include <cstddef>
#include <upcxx/upcxx.hpp>

#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "hash_map.hpp"

#include "butil.hpp"

int main(int argc, char **argv) {
  upcxx::init();

  // TODO: remove this, when you start writing
  // parallel implementation.
  // if (upcxx::rank_n() > 1) {
  //   throw std::runtime_error("Error: parallel implementation not started yet!"
  //     " (remove this when you start working.)");
  // }

  if (argc < 2) {
    BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test]\n");
    upcxx::finalize();
    exit(1);
  }

  std::string kmer_fname = std::string(argv[1]);
  std::string run_type = "";

  if (argc >= 3) {
    run_type = std::string(argv[2]);
  }

  int ks = kmer_size(kmer_fname);

  if (ks != KMER_LEN) {
    throw std::runtime_error("Error: " + kmer_fname + " contains " +
      std::to_string(ks) + "-mers, while this binary is compiled for " +
      std::to_string(KMER_LEN) + "-mers.  Modify packing.hpp and recompile.");
  }
  // Total number of kmers from all ranks.
  size_t n_kmers = line_count(kmer_fname);

  // Load factor of 0.5
  size_t hash_table_size = n_kmers * (1.0 / 0.5);

  size_t nprocs = upcxx::rank_n();
  size_t my_rank = upcxx::rank_me();
  // hash_table_size = (hash_table_size + nprocs - 1)/nprocs;

  HashMap hashmap(hash_table_size);

  // Build global pointer for data and used.

  if (run_type == "verbose") {
    BUtil::print("Initializing hash table of size %d for %d kmers.\n",
      hash_table_size, n_kmers);
  }

  std::vector <kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

  if (run_type == "verbose") {
    BUtil::print("Finished reading kmers.\n");
  }

  //
  // Building local hashmap and start nodes array.
  //
  auto start = std::chrono::high_resolution_clock::now();

  std::vector <kmer_pair> start_nodes;

  for (auto &kmer : kmers) {
    bool success = hashmap.insert(kmer);
    if (!success) {
      throw std::runtime_error("Error: HashMap is full!");
    }

    if (kmer.backwardExt() == 'F') {
      start_nodes.push_back(kmer);
    }
  }
  if (run_type == "verbose"){
    upcxx::barrier();
    int *lp = hashmap.used[upcxx::rank_me()].local();
    int count =0;
    for (int i=0;i<hashmap.my_size; i++){
      if(*(lp + i) != lp[i]) printf("errordsfadfasd\n");
      count+= lp[i]>0;
    }
    kmer_pair * kp = hashmap.data[upcxx::rank_me()].local();
    printf("my size: %d\n", hashmap.my_size);
    upcxx::barrier();
    printf(" Wrote %d in rank %d\n", count, upcxx::rank_me());
    upcxx::barrier();
    printf("First 3 elements in rank %d: %s = %d, %s = %d, %s = %d\n", upcxx::rank_me(), kp[0].kmer_str().c_str(), lp[0], kp[1].kmer_str().c_str(), lp[1],  kp[2].kmer_str().c_str(), lp[2]);
    
    // Write to file
    std::ofstream fout("verbose_" + std::to_string(upcxx::rank_me()) + ".dat");
    for (int i=0; i<hashmap.my_size; i++) {
      fout << kp[i].kmer_str().c_str() << ' ' << std::to_string(lp[i]) << std::endl;
    }
    fout.close();
  }
  auto end_insert = std::chrono::high_resolution_clock::now();
  
  // Build global pointer for used.


  // Measure time of buildinghash table.
  upcxx::barrier();
  double insert_time = std::chrono::duration <double> (end_insert - start).count();
  if (run_type != "test") {
    BUtil::print("Finished inserting in %lf\n", insert_time);
  }
  upcxx::barrier();

  //
  // building contigs
  //
  auto start_read = std::chrono::high_resolution_clock::now();
  std::list <std::list <kmer_pair>> contigs;
  for (const auto &start_kmer : start_nodes) {
    std::list <kmer_pair> contig;
    contig.push_back(start_kmer);
    while (contig.back().forwardExt() != 'F') {
      kmer_pair kmer;
      bool success = hashmap.find(contig.back().next_kmer(), kmer);
      if (!success) {
        printf("Doesn't find kmer str %s, end at kmer str %s \n", contig.back().next_kmer().get().c_str(), kmer.kmer_str().c_str());
        throw std::runtime_error("Error: k-mer not found in hashmap.");
      }
      contig.push_back(kmer);
    }
    contigs.push_back(contig);
  }

  auto end_read = std::chrono::high_resolution_clock::now();
  upcxx::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration <double> read = end_read - start_read;
  std::chrono::duration <double> insert = end_insert - start;
  std::chrono::duration <double> total = end - start;

  int numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
    [] (int sum, const std::list <kmer_pair> &contig) {
      return sum + contig.size();
    });

  if (run_type != "test") {
    BUtil::print("Assembled in %lf total\n", total.count());
  }

  if (run_type == "verbose") {
    printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
      " (%lf read, %lf insert, %lf total)\n", upcxx::rank_me(), contigs.size(),
      numKmers, start_nodes.size(), read.count(), insert.count(), total.count());
  }

  if (run_type == "test") {
    std::ofstream fout("test_" + std::to_string(upcxx::rank_me()) + ".dat");
    for (const auto &contig : contigs) {
      fout << extract_contig(contig) << std::endl;
    }
    fout.close();
  }

  upcxx::finalize();
  return 0;
}
