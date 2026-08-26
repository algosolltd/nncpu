/*
 * Copyright 2023 The ChampSim Contributors
 * Modifications copyright 2026 Algorithmica Solutions
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef NNCPU_REGULARITY_STRIDE_H
#define NNCPU_REGULARITY_STRIDE_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

#include "address.h"
#include "modules.h"
#include "msl/lru_table.h"

struct nncpu_regularity_stride : public champsim::modules::prefetcher {
  using delta_type = champsim::block_number::difference_type;

  struct tracker_entry {
    champsim::address ip{};
    champsim::block_number last_cl_addr{};
    delta_type last_stride{};

    auto index() const
    {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<2_b>();
    }

    auto tag() const
    {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<2_b>();
    }
  };

  struct lookahead_entry {
    champsim::address address{};
    champsim::address::difference_type stride{};
    int degree = 0;
  };

  static constexpr std::size_t DEFAULT_TRACKER_SETS = 256;
  static constexpr std::size_t DEFAULT_TRACKER_WAYS = 4;
  static constexpr std::size_t DEFAULT_WINDOW = 16;
  static constexpr std::size_t DEFAULT_WARMUP = 8;
  static constexpr int DEFAULT_SUPPORT_PERCENT = 35;
  static constexpr int DEFAULT_DEGREE = 3;

  champsim::modules::cache_module* cache_ = nullptr;
  bool gate_enabled = true;
  std::size_t window = DEFAULT_WINDOW;
  std::size_t warmup = DEFAULT_WARMUP;
  int support_percent = DEFAULT_SUPPORT_PERCENT;
  int prefetch_degree = DEFAULT_DEGREE;

  std::optional<champsim::block_number> last_global_block;
  std::deque<delta_type> recent_deltas;
  std::optional<lookahead_entry> active_lookahead;
  champsim::msl::lru_table<tracker_entry> table;

  std::uint64_t roi_decisions = 0;
  std::uint64_t roi_suppressed = 0;
  std::uint64_t roi_issued = 0;

  explicit nncpu_regularity_stride(champsim::modules::ModuleBuilder builder);

  void prefetcher_initialize() override {}
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                    bool cache_hit, bool useful_prefetch,
                                    access_type type, uint32_t metadata_in) override;
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way,
                                 bool prefetch, champsim::address evicted_addr,
                                 uint32_t metadata_in) override;
  void prefetcher_cycle_operate() override;
  void prefetcher_final_stats() override;
  void prefetcher_branch_operate(champsim::address, uint8_t,
                                 champsim::address) override {}

private:
  void observe(champsim::block_number block);
  bool gate_open() const;
};

#endif
