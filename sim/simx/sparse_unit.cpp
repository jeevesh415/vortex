
// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sparse_unit.h"
#include "sparse_cfg.h"
#include <rvfloats.h>
#include "core.h"
#include <cstring>

using namespace vortex;

namespace vt = vortex::sparse;
using cfg = vt::wmma_config_t<NUM_THREADS>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

template <typename It, typename Ot>
struct FMA {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static otype eval(itype a, itype b, otype c) {
    return static_cast<otype>(a) * static_cast<otype>(b) + c;
  }
};

template <>
struct FMA<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xc = rv_htof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftoh_s(xd, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::bf16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::bf16, vt::bf16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xc = rv_btof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftob_s(xd, 0, nullptr);
    return xh;
  }
};

template <typename It, typename Ot>
struct FEDP {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
  constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
  static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
  auto acc = bit_cast<otype>(c_val);
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
    auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA<It, Ot>::eval(a[i], b[i], acc);
    }
  }
  return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::int4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        if (a_val & 0x8) {
          a_val |= 0xFFFFFFF0;
        }
        if (b_val & 0x8) {
          b_val |= 0xFFFFFFF0;
        }
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::uint4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

// Sparse FEDP: uses metadata to select which values from fragB to use
// fragA is sparse (2:4), fragB is dense
// metadata contains bitmasks indicating which 2 of 4 positions are non-zero
// Each metadata byte is a bitmask where bits 0-3 indicate positions (0-3) that are kept
// For fp16/bf16: each register has 2 elements, metadata byte covers 4 elements = 2 registers
// For fp32: each register has 1 element, metadata byte covers 4 elements = 4 registers
template <typename It, typename Ot>
struct SparseFEDP {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, const uint32_t* metadata) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "SparseFEDP: tcK * i_ratio must be <= 32");
    auto acc = bit_cast<otype>(c_val);
    
    // Process registers in blocks of 4 elements
    // Each metadata byte covers 4 elements: for fp16 that's 2 registers, for fp32 that's 4 registers
    constexpr uint32_t regs_per_block = (i_ratio == 2) ? 2 : 4;
    
    for (uint32_t z = 0; z < cfg::tcK; z += regs_per_block) {
      // Get metadata for this block of 4 elements
      uint32_t block_idx = z / regs_per_block;
      uint32_t meta = (block_idx < 8) ? metadata[block_idx] : 0;
      uint8_t meta_byte = meta & 0xFF;
      
      // Process each of the 4 positions in this block
      for (uint32_t pos = 0; pos < 4; ++pos) {
        if (meta_byte & (1u << pos)) {
          // This position is non-zero, find which register and element it corresponds to
          uint32_t reg_idx = z + (pos / i_ratio);
          uint32_t elem_idx = pos % i_ratio;
          
          if (reg_idx < cfg::tcK) {
            auto a = reinterpret_cast<const itype *>(&a_row[reg_idx].u32);
            auto b = reinterpret_cast<const itype *>(&b_col[reg_idx].u32);
            acc = FMA<It, Ot>::eval(a[elem_idx], b[elem_idx], acc);
          }
        }
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

// Specialized version for fp32->fp32: each register contains 1 element
// Metadata byte encodes which 2 of 4 consecutive elements (across 4 registers) are non-zero
// The metadata byte is a bitmask where bits 0-3 indicate which positions (0-3) are kept
template <>
struct SparseFEDP<vt::fp32, vt::fp32> {
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, const uint32_t* metadata) {
    auto acc = bit_cast<float>(c_val);
    
    // Process registers in groups of 4 (one block of 4 elements = 4 registers)
    for (uint32_t z = 0; z < cfg::tcK; z += 4) {
      // Calculate which block this is (each block = 4 registers = 4 elements)
      uint32_t block_idx = z / 4;
      
      // Get metadata for this block
      // Each metadata uint32_t contains 4 bytes (one per block of 4 elements)
      uint32_t meta_reg_idx = block_idx / 4;  // Which metadata register (0-7)
      if (meta_reg_idx >= 8) break;  // Only 8 metadata registers available
      
      uint32_t meta = metadata[meta_reg_idx];
      // Extract the byte for this block (each uint32_t has 4 bytes)
      uint32_t byte_offset = block_idx % 4;
      uint8_t meta_byte = (meta >> (byte_offset * 8)) & 0xFF;
      
      // Decode which 2 of 4 positions are non-zero
      // The byte is a bitmask where bits 0-3 indicate which positions (0-3) are kept
      // For 2:4 sparsity, exactly 2 bits should be set
      for (uint32_t i = 0; i < 4 && (z + i) < cfg::tcK; ++i) {
        if (meta_byte & (1u << i)) {
          // This position is non-zero, multiply and accumulate
          auto a_val = bit_cast<float>(a_row[z + i].u32);
          auto b_val = bit_cast<float>(b_col[z + i].u32);
          acc = FMA<vt::fp32, vt::fp32>::eval(a_val, b_val, acc);
        }
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

using PFN_FEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t);
using PFN_SparseFEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t, const uint32_t*);

static PFN_SparseFEDP select_SparseFEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp32::id:
      return SparseFEDP<vt::fp32, vt::fp32>::eval;
    case vt::fp16::id:
      return SparseFEDP<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return SparseFEDP<vt::bf16, vt::fp32>::eval;
    default:
      std::cout << "Error: unsupported sparse mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return SparseFEDP<vt::fp16, vt::fp16>::eval;
    default:
      std::cout << "Error: unsupported sparse mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return SparseFEDP<vt::bf16, vt::bf16>::eval;
    default:
      std::cout << "Error: unsupported sparse mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported sparse output type: " << OT << "!" << std::endl;
    std::abort();
  }
}

static PFN_FEDP select_FEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::fp32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::bf16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::int32::id:
    switch (IT) {
    case vt::int8::id:
      return FEDP<vt::int8, vt::int32>::eval;
    case vt::uint8::id:
      return FEDP<vt::uint8, vt::int32>::eval;
    case vt::int4::id:
      return FEDP<vt::int4, vt::int32>::eval;
    case vt::uint4::id:
      return FEDP<vt::uint4, vt::int32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported output type: " << OT << "!" << std::endl;
    std::abort();
  }
}

class SparseUnit::Impl {
public:
  Impl(SparseUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
    , tile_reg_file_(8, std::vector<std::vector<typename vt::fp32::dtype>>(16, std::vector<typename vt::fp32::dtype>(32, 0.0f)))
    , metadata_reg_file_(8, std::vector<std::vector<typename vt::uint4::dtype>>(16, std::vector<typename vt::uint4::dtype>(32, 0)))
  {
    // Register file initialized: 8 registers, each 16x32 fp32 elements
  }

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
    // Reset tile register file to zero
    for (auto& reg : tile_reg_file_) {
      for (auto& row : reg) {
        std::fill(row.begin(), row.end(), 0.0f);
      }
    }
    // Reset metadata register file to zero
    for (auto& reg : metadata_reg_file_) {
      for (auto& row : reg) {
        std::fill(row.begin(), row.end(), 0);
      }
    }
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.front();
      int delay = 0;
      auto tcu_type = std::get<VegetaTcuType>(trace->op_type);
      switch (tcu_type) {
      case VegetaTcuType::TILE_GEMM_T:
      case VegetaTcuType::TILE_GEMM_U:
      case VegetaTcuType::TILE_GEMM_V:
      case VegetaTcuType::TILE_GEMM_R:
      case VegetaTcuType::WMMA:
        delay = 4;
        break;
      default:
        std::abort();
      }
      DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      input.pop();
    }
  }

  void wmma(uint32_t wid,
            uint32_t fmt_s,
            uint32_t fmt_d,
            uint32_t step_m,
            uint32_t step_n,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data,
            const uint32_t* metadata) {
    __unused(trace_data);

    // Use provided metadata from integer registers 0-7 for sparse fragA
    // If metadata is null, use zeros (dense mode fallback)
    uint32_t meta[8] = {0};
    if (metadata != nullptr) {
      for (uint32_t i = 0; i < 8; ++i) {
        meta[i] = metadata[i];
      }
    }
    
    // Use sparse FEDP for sparse-dense GEMM
    auto sparse_fedp = select_SparseFEDP(fmt_s, fmt_d);

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (step_n % cfg::b_sub_blocks) * cfg::b_block_size;

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        
        uint32_t idx = i * cfg::tcN + j;
        if (idx >= rs3_data.size() || idx >= rd_data.size()) {
          std::cout << "Error: index out of bounds in sparse_unit wmma: idx=" << idx 
                    << ", rs3_data.size()=" << rs3_data.size() 
                    << ", rd_data.size()=" << rd_data.size() << std::endl;
          std::abort();
        }
        
        auto c_val = rs3_data.at(idx).u32;
        
        // Map metadata from fragment registers to K dimension registers
        // Fragment register r maps to: block_m = r / cfg::k_steps, block_k = r % cfg::k_steps
        // For K dimension register z, we need fragment register r where block_k corresponds to z
        uint32_t meta_for_k[8] = {0};
        for (uint32_t z = 0; z < cfg::tcK && z < 8; ++z) {
          // Compute which fragment register contains data for K dimension z and M dimension i
          // Fragment register index = a_off + i * cfg::tcK + z
          uint32_t frag_reg_idx = a_off + i * cfg::tcK + z;
          if (frag_reg_idx < 8) {
            meta_for_k[z] = meta[frag_reg_idx];
          }
        }
        
        // Perform sparse-dense FEDP: fragA is sparse, fragB is dense
        // Use metadata to select which values from fragB to multiply
        auto d_val = sparse_fedp(a_row, b_col, c_val, meta_for_k);
        rd_data.at(idx).u64 = nan_box(d_val);

        DTH(3, "SparseFEDP: wid=" << wid << ", i=" << i << ", j=" << j << ", m=" << step_m << ", n=" << step_n << ", a_row={" << std::hex);
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << a_row[q].u32);
        }
        DTN(3, "}, b_col={");
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << b_col[q].u32);
        }
        DTN(3, "}, metadata={");
        for (uint32_t q = 0; q < 8 && q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << meta_for_k[q]);
        }
        DTN(3, "}, c_val=0x" << c_val << ", d_val=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  // Map ureg index to tile register indices
  // ureg 0 -> tile reg 0, 1
  // ureg 1 -> tile reg 2, 3
  // ureg 2 -> tile reg 4, 5
  // etc.
  static std::vector<uint32_t> map_ureg_to_treg(uint32_t ureg_idx) {
    std::vector<uint32_t> treg_indices;
    treg_indices.push_back(ureg_idx * 2);
    treg_indices.push_back(ureg_idx * 2 + 1);
    return treg_indices;
  }

  // Map vreg index to tile register indices
  // vreg 0 -> tile reg 0, 1, 2, 3
  // vreg 1 -> tile reg 4, 5, 6, 7
  // etc.
  static std::vector<uint32_t> map_vreg_to_treg(uint32_t vreg_idx) {
    std::vector<uint32_t> treg_indices;
    treg_indices.push_back(vreg_idx * 4);
    treg_indices.push_back(vreg_idx * 4 + 1);
    treg_indices.push_back(vreg_idx * 4 + 2);
    treg_indices.push_back(vreg_idx * 4 + 3);
    return treg_indices;
  }

  void load(const Instr &instr,
            uint32_t wid,
            uint32_t tid,
            const std::vector<reg_data_t> &rs1_data,
            MemTraceData *trace_data) {
    __unused(wid);
    auto lsu_type = std::get<VegetaLsuType>(instr.getOpType());
    auto lsuArgs = std::get<IntrVegetaLsuArgs>(instr.getArgs());
    uint32_t vd = instr.getDestReg().idx; // DestReg contains the tile register index
    
    // Calculate base address: rs1_data + immediate offset
    uint64_t base_addr = rs1_data.at(tid).i + lsuArgs.offset;

    constexpr uint32_t TILE_ROWS = 16;
    constexpr uint32_t TILE_COLS = 32;

    switch (lsu_type) {
    case VegetaLsuType::TILE_LOAD_T: {
      // tile_load_t: store in tile register specified by DestReg
      uint32_t tile_reg_idx = vd;
      assert(tile_reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
      auto &tile_reg = tile_reg_file_[tile_reg_idx];
      constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype); // 4 bytes for fp32
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads

      // Load tile from memory: 16 rows x 32 columns = 512 fp32 elements = 2048 bytes
      for (uint32_t row = 0; row < TILE_ROWS; ++row) {
        for (uint32_t col = 0; col < TILE_COLS; ++col) {
          uint64_t mem_addr = base_addr + (row * TILE_COLS + col) * ELEMENT_SIZE;
          uint32_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, ELEMENT_SIZE);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
          
          // Interpret as float and store in tile register
          float value;
          std::memcpy(&value, &mem_data, ELEMENT_SIZE);
          tile_reg[row][col] = value;
        }
      }
      
      DP(2, "TILE_LOAD_T: wid=" << wid << ", tid=" << tid 
         << ", tile_reg_idx=" << tile_reg_idx << ", base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_U: {
      // tile_load_u: DestReg contains ureg index, map to tile registers
      // ureg 0 -> tile reg 0, 1
      std::vector<uint32_t> target_tregs = map_ureg_to_treg(vd);
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads
      constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype);
      
      for (uint32_t treg_idx : target_tregs) {
        assert(treg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
        auto &tile_reg = tile_reg_file_[treg_idx];
        
        // Load tile from memory: 16 rows x 32 columns = 512 fp32 elements = 2048 bytes
        for (uint32_t row = 0; row < TILE_ROWS; ++row) {
          for (uint32_t col = 0; col < TILE_COLS; ++col) {
            uint64_t mem_addr = base_addr + (row * TILE_COLS + col) * ELEMENT_SIZE;
            uint32_t mem_data = 0;
            core_->dcache_read(&mem_data, mem_addr, ELEMENT_SIZE);
            trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
            
            float value;
            std::memcpy(&value, &mem_data, ELEMENT_SIZE);
            tile_reg[row][col] = value;
          }
        }
      }
      
      DP(2, "TILE_LOAD_U: wid=" << wid << ", tid=" << tid 
         << ", ureg_idx=" << vd << ", target_tregs=[" 
         << target_tregs[0] << ", " << target_tregs[1] << "], base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_V: {
      // tile_load_v: DestReg contains vreg index, map to tile registers
      // vreg 0 -> tile reg 0, 1, 2, 3
      std::vector<uint32_t> target_tregs = map_vreg_to_treg(vd);
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads
      constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype);
      
      for (uint32_t treg_idx : target_tregs) {
        assert(treg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
        auto &tile_reg = tile_reg_file_[treg_idx];
        
        // Load tile from memory: 16 rows x 32 columns = 512 fp32 elements = 2048 bytes
        for (uint32_t row = 0; row < TILE_ROWS; ++row) {
          for (uint32_t col = 0; col < TILE_COLS; ++col) {
            uint64_t mem_addr = base_addr + (row * TILE_COLS + col) * ELEMENT_SIZE;
            uint32_t mem_data = 0;
            core_->dcache_read(&mem_data, mem_addr, ELEMENT_SIZE);
            trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
            
            float value;
            std::memcpy(&value, &mem_data, ELEMENT_SIZE);
            tile_reg[row][col] = value;
          }
        }
      }
      
      DP(2, "TILE_LOAD_V: wid=" << wid << ", tid=" << tid 
         << ", vreg_idx=" << vd << ", target_tregs=[" 
         << target_tregs[0] << ", " << target_tregs[1] << ", " 
         << target_tregs[2] << ", " << target_tregs[3] << "], base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_M: {
      // tile_load_M: DestReg contains metadata register index, store in that metadata register
      uint32_t meta_reg_idx = vd;
      assert(meta_reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
      auto &metadata_reg = metadata_reg_file_[meta_reg_idx];
      constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::uint4::dtype); // 1 byte for uint8_t (stores one uint4)

      // Load metadata from memory: 16 rows x 32 columns = 512 uint4 elements = 512 bytes
      // Note: Each uint4 is stored in the lower 4 bits of a byte
      for (uint32_t row = 0; row < TILE_ROWS; ++row) {
        for (uint32_t col = 0; col < TILE_COLS; ++col) {
          uint64_t mem_addr = base_addr + (row * TILE_COLS + col) * ELEMENT_SIZE;
          uint8_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, ELEMENT_SIZE);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
          
          // Store only lower 4 bits (uint4 value)
          metadata_reg[row][col] = mem_data & 0x0F;
        }
      }

      DP(2, "TILE_LOAD_M: wid=" << wid << ", tid=" << tid 
         << ", metadata_reg_idx=" << meta_reg_idx << ", base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    default:
      std::abort();
    }
  }

  void store(const Instr &instr,
             uint32_t wid,
             uint32_t tid,
             const std::vector<reg_data_t> &rs1_data,
             MemTraceData *trace_data) {
    __unused(wid);

    auto lsuArgs = std::get<IntrVegetaLsuArgs>(instr.getArgs());
    uint32_t vs3 = instr.getSrcReg(1).idx; // Source tile register index
    
    // Calculate base address: rs1_data + immediate offset
    uint64_t base_addr = rs1_data.at(tid).i + lsuArgs.offset;
    base_addr &= 0xFFFFFFFC; // Align to word boundary

    assert(vs3 < tile_reg_file_.size() && "Tile register index out of bounds");
    auto &tile_reg = tile_reg_file_[vs3];
    constexpr uint32_t TILE_ROWS = 16;
    constexpr uint32_t TILE_COLS = 32;
    constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype); // 4 bytes for fp32

    // Store tile to memory: 16 rows x 32 columns = 512 fp32 elements = 2048 bytes
    for (uint32_t row = 0; row < TILE_ROWS; ++row) {
      for (uint32_t col = 0; col < TILE_COLS; ++col) {
        uint64_t mem_addr = base_addr + (row * TILE_COLS + col) * ELEMENT_SIZE;
        float value = tile_reg[row][col];
        uint32_t mem_data = 0;
        std::memcpy(&mem_data, &value, ELEMENT_SIZE);
        core_->dcache_write(&mem_data, mem_addr, ELEMENT_SIZE);
        trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
      }
    }

    DP(2, "TILE_STORE: wid=" << wid << ", tid=" << tid << ", vs3=" << vs3 
       << ", base_addr=0x" << std::hex << base_addr << std::dec);
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

  // Tile register file accessors
  SparseRegFile_t& tile_reg_file() {
    return tile_reg_file_;
  }

  const SparseRegFile_t& tile_reg_file() const {
    return tile_reg_file_;
  }

  // Metadata register file accessors
  std::vector<std::vector<std::vector<typename vt::uint4::dtype>>>& metadata_reg_file() {
    return metadata_reg_file_;
  }

  const std::vector<std::vector<std::vector<typename vt::uint4::dtype>>>& metadata_reg_file() const {
    return metadata_reg_file_;
  }

  // Access a specific tile register (returns reference to 16x32 vector)
  std::vector<std::vector<typename vt::fp32::dtype>>& get_tile_reg(uint32_t reg_idx) {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    return tile_reg_file_[reg_idx];
  }

  const std::vector<std::vector<typename vt::fp32::dtype>>& get_tile_reg(uint32_t reg_idx) const {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    return tile_reg_file_[reg_idx];
  }

  // Access a specific metadata register (returns reference to 16x32 vector)
  std::vector<std::vector<typename vt::uint4::dtype>>& get_metadata_reg(uint32_t reg_idx) {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    return metadata_reg_file_[reg_idx];
  }

  const std::vector<std::vector<typename vt::uint4::dtype>>& get_metadata_reg(uint32_t reg_idx) const {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    return metadata_reg_file_[reg_idx];
  }

  // Access a specific element in a tile register
  typename vt::fp32::dtype& get_tile_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    assert(row < tile_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < tile_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return tile_reg_file_[reg_idx][row][col];
  }

  const typename vt::fp32::dtype& get_tile_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) const {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    assert(row < tile_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < tile_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return tile_reg_file_[reg_idx][row][col];
  }

  // Access a specific element in a metadata register
  typename vt::uint4::dtype& get_metadata_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    assert(row < metadata_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < metadata_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return metadata_reg_file_[reg_idx][row][col];
  }

  const typename vt::uint4::dtype& get_metadata_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) const {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    assert(row < metadata_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < metadata_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return metadata_reg_file_[reg_idx][row][col];
  }

private:

  SparseUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
  SparseRegFile_t tile_reg_file_;  // 8 registers, each 16x32 fp32 elements
  std::vector<std::vector<std::vector<typename vt::uint4::dtype>>> metadata_reg_file_;  // 8 registers, each 16x32 uint4 elements
};

///////////////////////////////////////////////////////////////////////////////

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n), ""};
  default:
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////

SparseUnit::SparseUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
	: SimObject<SparseUnit>(ctx, name)
	, Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, core))
{}

SparseUnit::~SparseUnit() {
  delete impl_;
}

void SparseUnit::reset() {
  impl_->reset();
}

void SparseUnit::tick() {
  impl_->tick();
}

const SparseUnit::PerfStats &SparseUnit::perf_stats() const {
	return impl_->perf_stats();
}

void SparseUnit::load(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, MemTraceData* trace_data) {
  impl_->load(instr, wid, tid, rs1_data, trace_data);
}

void SparseUnit::store(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, MemTraceData* trace_data) {
  impl_->store(instr, wid, tid, rs1_data, trace_data);
}

void SparseUnit::wmma(uint32_t wid,
                      uint32_t fmt_s,
                      uint32_t fmt_d,
                      uint32_t step_m,
                      uint32_t step_n,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data,
                      const uint32_t* metadata) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data, metadata);
}
