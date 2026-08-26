/*
 * Copyright 2023 The ChampSim Contributors
 * Modifications copyright 2026 Algorithmica Solutions
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */

#include "nncpu_regularity_stride.h"

#include <algorithm>
#include <stdexcept>

#include <fmt/core.h>

champsim::modules::prefetcher::register_module<nncpu_regularity_stride>
    nncpu_regularity_stride_register("nncpu_regularity_stride");

nncpu_regularity_stride::nncpu_regularity_stride(
    champsim::modules::ModuleBuilder builder)
    : cache_(builder.get_parent<champsim::modules::cache_module>()),
      gate_enabled(builder.get_parameter<bool>("gate_enabled", true, true)),
      window(builder.get_parameter<std::size_t>("window", true, DEFAULT_WINDOW)),
      warmup(builder.get_parameter<std::size_t>("warmup", true, DEFAULT_WARMUP)),
      support_percent(builder.get_parameter<int>("support_percent", true,
                                                 DEFAULT_SUPPORT_PERCENT)),
      prefetch_degree(builder.get_parameter<int>("degree", true, DEFAULT_DEGREE)),
      table(builder.get_parameter<std::size_t>("tracker_sets", true,
                                               DEFAULT_TRACKER_SETS),
            builder.get_parameter<std::size_t>("tracker_ways", true,
                                               DEFAULT_TRACKER_WAYS))
{
  if (window == 0 || warmup > window)
    throw std::invalid_argument("regularity window must be positive and >= warmup");
  if (support_percent < 0 || support_percent > 100)
    throw std::invalid_argument("support_percent must be within [0, 100]");
  if (prefetch_degree <= 0)
    throw std::invalid_argument("degree must be positive");
}

void nncpu_regularity_stride::observe(champsim::block_number block)
{
  if (last_global_block.has_value()) {
    recent_deltas.push_back(champsim::offset(*last_global_block, block));
    while (recent_deltas.size() > window)
      recent_deltas.pop_front();
  }
  last_global_block = block;
}

bool nncpu_regularity_stride::gate_open() const
{
  if (!gate_enabled || recent_deltas.size() < warmup)
    return true;

  std::size_t mode_count = 0;
  for (auto candidate : recent_deltas) {
    const auto count = static_cast<std::size_t>(
        std::count(recent_deltas.begin(), recent_deltas.end(), candidate));
    mode_count = std::max(mode_count, count);
  }
  return mode_count * 100 >=
         static_cast<std::size_t>(support_percent) * recent_deltas.size();
}

uint32_t nncpu_regularity_stride::prefetcher_cache_operate(
    champsim::address addr, champsim::address ip, bool cache_hit,
    bool useful_prefetch, access_type type, uint32_t metadata_in)
{
  (void)cache_hit;
  (void)useful_prefetch;
  if (type != access_type::LOAD && type != access_type::RFO)
    return metadata_in;

  const champsim::block_number block{addr};
  observe(block);
  if (!cache_->warmup)
    ++roi_decisions;

  delta_type stride = 0;
  auto found = table.check_hit({ip, block, stride});
  if (found.has_value()) {
    stride = champsim::offset(found->last_cl_addr, block);
    if (stride != 0 && stride == found->last_stride) {
      if (gate_open())
        active_lookahead = {champsim::address{block}, stride, prefetch_degree};
      else {
        active_lookahead.reset();
        if (!cache_->warmup)
          ++roi_suppressed;
      }
    }
  }
  table.fill({ip, block, stride});
  return metadata_in;
}

void nncpu_regularity_stride::prefetcher_cycle_operate()
{
  if (!active_lookahead.has_value())
    return;
  if (!gate_open()) {
    active_lookahead.reset();
    return;
  }

  auto [old_address, stride, degree] = *active_lookahead;
  const champsim::address pf_address{
      champsim::block_number{old_address} + stride};
  if (!cache_->is_virtual_prefetch() &&
      champsim::page_number{pf_address} != champsim::page_number{old_address}) {
    active_lookahead.reset();
    return;
  }

  const bool fill_l1 = cache_->get_mshr_occupancy_ratio() < 0.5;
  if (prefetch_line(pf_address, fill_l1, 0)) {
    if (!cache_->warmup)
      ++roi_issued;
    --degree;
    if (degree == 0)
      active_lookahead.reset();
    else
      active_lookahead = {pf_address, stride, degree};
  }
}

uint32_t nncpu_regularity_stride::prefetcher_cache_fill(
    champsim::address addr, long set, long way, bool prefetch,
    champsim::address evicted_addr, uint32_t metadata_in)
{
  (void)addr;
  (void)set;
  (void)way;
  (void)prefetch;
  (void)evicted_addr;
  return metadata_in;
}

void nncpu_regularity_stride::prefetcher_final_stats()
{
  fmt::print("NNCPU_REGULARITY roi_decisions={} roi_suppressed={} roi_issued={} "
             "gate_enabled={} window={} support_percent={} degree={}\n",
             roi_decisions, roi_suppressed, roi_issued, gate_enabled, window,
             support_percent, prefetch_degree);
}
