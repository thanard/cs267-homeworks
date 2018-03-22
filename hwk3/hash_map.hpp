#pragma once

#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"
#include <atomic>

struct HashMap {
  // std::vector <kmer_pair> data;
  // std::vector <int> used;

  // upcxx::global_ptr<std::vector <kmer_pair>> gdataptr = nullptr;
  // if (upcxx::rank_me() == 0) {
  //   gdataptr = upcxx::new_array<std::vector <kmer_pair>>(upcxx::rank_n());
  // }
  // gdataptr = upcxx::broadcast(gdataptr, 0).wait();

  // upcxx::global_ptr<std::vector <kmer_pair>> my_gdataptr = gdataptr + upcxx::rank_me();

  // upcxx::rput(hashmap.data, my_gdataptr);
  // upcxx::dist_object<HashMap> hashmap(hash_table_size);

  // data_ptr[my_rank] = //point to global data variable but local chunk
  // used_ptr[my_rank] = //point to global used

  int nprocs;
  int my_rank;

  size_t my_size;
  size_t global_size;

  size_t size() const noexcept;

  HashMap(size_t size);

  // Most important functions: insert and retrieve
  // k-mers from the hash table.
  bool insert(const kmer_pair &kmer);
  bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

  // Helper functions

  // Write and read to a logical data slot in the table.
  void write_slot(uint64_t slot, const kmer_pair &kmer);
  kmer_pair read_slot(uint64_t slot);

  // Request a slot or check if it's already used.
  bool request_slot(uint64_t slot);
  bool slot_used(uint64_t slot);

  std::vector<upcxx::global_ptr<kmer_pair>> data;
  std::vector<upcxx::global_ptr<int>> used;

  int which_rank(uint64_t slot);
};

HashMap::HashMap(size_t size) {
  nprocs = upcxx::rank_n();
  my_rank = upcxx::rank_me();
  my_size = size_t ((size + nprocs -1 )/ nprocs);
  global_size = size;

  data.resize(nprocs);
  used.resize(nprocs);
  data[upcxx::rank_me()] = upcxx::new_array<kmer_pair>(my_size);
  used[upcxx::rank_me()] = upcxx::new_array<int>(my_size);
  for (int i=0; i<nprocs; i++){
    data[i] = upcxx::broadcast(data[i],i).wait();
    used[i] = upcxx::broadcast(used[i],i).wait();
  }

}

// TODO: rput(local kmer_pair, gdataptr)
// size is already global
// find which rank it needs to write to.
bool HashMap::insert(const kmer_pair &kmer) {
  uint64_t hash = kmer.hash();
  uint64_t probe = 0;
  bool success = false;

  do {
    uint64_t slot = (hash + probe++) % global_size;
    success = request_slot(slot);
    if (success) {
      write_slot(slot, kmer);
    }
  } while (!success && probe < global_size);
  return success;
}

// TODO: rget(gdataptr, )
// How do we know which rank we need to look up for a specific hash?
bool HashMap::find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
  uint64_t hash = key_kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % global_size;
    if (slot_used(slot)) {
    // if (true){
      val_kmer = read_slot(slot);
      if (val_kmer.kmer == key_kmer) {
        success = true;
      }
    }
  } while (!success && probe < global_size);
  if (!success){
    uint64_t slot = (hash + probe) % global_size;
    printf("global_size: %d, probe: %d\n", global_size, probe);
    printf("global slot: %d, slot rank: %d, local slot: %d\n", slot, which_rank(slot), slot % my_size);
  }
  return success;
}

bool HashMap::slot_used(uint64_t slot) {
  // return used[slot] != 0;
  upcxx::future<int> local_used = upcxx::rget(used[which_rank(slot)] + slot % my_size);
  local_used.wait();
  return 0 != local_used.result();
}

void HashMap::write_slot(uint64_t slot, const kmer_pair &kmer) {
  // data[slot] = kmer;
  upcxx::rput(kmer, data[which_rank(slot)] + slot % my_size).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) {
  // return data[slot];
  upcxx::future<kmer_pair> local_data = upcxx::rget(data[which_rank(slot)] + slot % my_size);
  local_data.wait();
  return local_data.result();
}

bool HashMap::request_slot(uint64_t slot) {
  // upcxx::future<int> local_used = upcxx::atomic_get(used[which_rank(slot)] + slot % my_size, std::memory_order_relaxed);
  upcxx::future<int> local_used = upcxx::atomic_fetch_add(used[which_rank(slot)] + slot % my_size, 1, std::memory_order_relaxed);
  // if (used[slot] != 0) {
  local_used.wait();
  if (local_used.result() != 0){ 
    return false;
  } else {
//    used[slot] = 1;
    // upcxx::atomic_put(used[which_rank(slot)] + slot % my_size, 1, std::memory_order_relaxed).wait();
    // upcxx::rput(1, used[which_rank(slot)] + slot % my_size);
    return true;
  }
}

// size_t HashMap::global_size() const noexcept {
//   return global_size;
// }

int HashMap::which_rank(uint64_t slot) {
  if (slot >= global_size || slot <0){
    throw std::runtime_error("Error: input has to be in [0, global_size-1]. ");
  }
  return int(slot / my_size);
}