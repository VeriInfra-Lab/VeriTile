import bench.tritonbench_g.chunk_gate_recurrence.Kernels
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
open VeriTile.Triton
namespace Old
export VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource (forward backward)
end Old

/-- Frozen original source, retained separately from the repair. -/
def originalForward := Old.forward .fp32 .fp32 .fp32

/-- Complete repaired FP32 forward source. The four seed strides are the
only kernel change; the N=0 branch is a host action. All casts are retained. -/
def forward (S D O seed : RegionName) (H N K V SB SH SK SV : Nat) (hasSeed : Bool) :=
  triton {
    tl.ports inputs=[S,D,seed], outputs=[O]
    tl.region S=tl.float32, D=tl.float32, O=tl.float32, seed=tl.float32
    h = tl.program_id(0)
    tk = tl.program_id(1)
    tv = tl.program_id(2)
    S = S + h * $(N) * $(K) * $(V) + tk * $(V) * $(64) +
      tl.arange(0, $(64))[:, None] * $(V) + tv * $(16) + tl.arange(0, $(16))[None, :]
    O = O + h * $(N) * $(K) * $(V) + tk * $(V) * $(64) +
      tl.arange(0, $(64))[:, None] * $(V) + tv * $(16) + tl.arange(0, $(16))[None, :]
    if hasSeed {
      seed = seed + (h // $(H)) * $(SB) + (h % $(H)) * $(SH) +
        (tk * $(64) + tl.arange(0, $(64))[:, None]) * $(SK) +
        (tv * $(16) + tl.arange(0, $(16))[None, :]) * $(SV)
      acc = tl.load(seed).to(tl.float32)
    } else {
      acc = tl.zeros([$(64), $(16)], dtype=tl.float32)
    }
    tl.store(O, (acc).to(tl.float32))
    O += $(K) * $(V)
    D = D + h * $(N)
    for _i in range($(0), $(N)-$(1), $(1)) {
      d_i = tl.load(D)
      S_i = tl.load(S)
      acc = acc*d_i+S_i
      tl.store(O, (acc).to(tl.float32))
      D += $(1)
      S += $(K)*$(V)
      O += $(K)*$(V)
    }
  }

/-- Existing addition-order candidate with the repaired four-stride seed load. -/
def forwardSumLeft (S D O seed : RegionName) (H N K V SB SH SK SV : Nat) (hasSeed : Bool) :=
  triton {
    tl.ports inputs=[S,D,seed], outputs=[O]
    tl.region S=tl.float32, D=tl.float32, O=tl.float32, seed=tl.float32
    h = tl.program_id(0)
    tk = tl.program_id(1)
    tv = tl.program_id(2)
    S = S + h * $(N) * $(K) * $(V) + tk * $(V) * $(64) +
      tl.arange(0, $(64))[:, None] * $(V) + tv * $(16) + tl.arange(0, $(16))[None, :]
    O = O + h * $(N) * $(K) * $(V) + tk * $(V) * $(64) +
      tl.arange(0, $(64))[:, None] * $(V) + tv * $(16) + tl.arange(0, $(16))[None, :]
    if hasSeed {
      seed = seed + (h // $(H)) * $(SB) + (h % $(H)) * $(SH) +
        (tk * $(64) + tl.arange(0, $(64))[:, None]) * $(SK) +
        (tv * $(16) + tl.arange(0, $(16))[None, :]) * $(SV)
      acc = tl.load(seed).to(tl.float32)
    } else {
      acc = tl.zeros([$(64), $(16)], dtype=tl.float32)
    }
    tl.store(O, (acc).to(tl.float32))
    O += $(K) * $(V)
    D = D + h * $(N)
    for _i in range($(0), $(N)-$(1), $(1)) {
      d_i = tl.load(D)
      S_i = tl.load(S)
      acc = S_i+acc*d_i
      tl.store(O, (acc).to(tl.float32))
      D += $(1)
      S += $(K)*$(V)
      O += $(K)*$(V)
    }
  }

/-- Complete unchanged FP32 backward body, including DG and final DL. -/
def backward (S D DI DG DL DS : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :=
  let dtype : FloatDType := .fp32
  triton {
    tl.ports inputs=[S, D, DS], outputs=[DI, DG, DL]
    tl.region S = DTYPE, DL = DTYPE, D = tl.float32, DG = tl.float32,
      DS = tl.float32, DI = tl.float32
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    NUM_K = $(D_MODEL_K) // $(BLOCK_MODEL_K)
    NUM_V = $(D_MODEL_V) // $(BLOCK_MODEL_V)
    offs_k = tl.arange(0, $(BLOCK_MODEL_K))
    offs_v = tl.arange(0, $(BLOCK_MODEL_V))
    S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
    DI = DI + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
    DS = DS + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :] + ($(NUM_BLOCK) - $(1)) * $(D_MODEL_K) * $(D_MODEL_V)
    DG = DG + offset_bh * $(NUM_BLOCK) * NUM_K * NUM_V +
      offset_d * NUM_V + offset_s + ($(NUM_BLOCK) - $(2)) * NUM_K * NUM_V
    D = D + offset_bh * $(NUM_BLOCK) + ($(NUM_BLOCK) - $(1))
    Dacc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
    for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
      S_i = tl.load(S)
      DS_i = tl.load(DS)
      d_i = tl.load(D)
      Dacc = Dacc * d_i + DS_i
      DG_i = tl.sum(Dacc * (S_i).to(tl.float32))
      tl.store(DG, (DG_i).to(tl.float32))
      tl.store(DI, (Dacc).to(tl.float32))
      S -= $(D_MODEL_K) * $(D_MODEL_V)
      DI -= $(D_MODEL_K) * $(D_MODEL_V)
      DS -= $(D_MODEL_K) * $(D_MODEL_V)
      DG -= NUM_K * NUM_V
      D -= $(1)
    }
    DL = DL + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :]
    DS_i = tl.load(DS)
    d_i = tl.load(D)
    Dacc = Dacc * d_i + DS_i
    tl.store(DL, (Dacc).to(DTYPE))
  }

/-- Existing backward addition-order candidate; pointers, reductions and stores are unchanged. -/
def backwardSumLeft (S D DI DG DL DS : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :=
  let dtype : FloatDType := .fp32
  triton {
    tl.ports inputs=[S, D, DS], outputs=[DI, DG, DL]
    tl.region S = DTYPE, DL = DTYPE, D = tl.float32, DG = tl.float32,
      DS = tl.float32, DI = tl.float32
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    NUM_K = $(D_MODEL_K) // $(BLOCK_MODEL_K)
    NUM_V = $(D_MODEL_V) // $(BLOCK_MODEL_V)
    offs_k = tl.arange(0, $(BLOCK_MODEL_K))
    offs_v = tl.arange(0, $(BLOCK_MODEL_V))
    S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
    DI = DI + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
    DS = DS + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :] + ($(NUM_BLOCK) - $(1)) * $(D_MODEL_K) * $(D_MODEL_V)
    DG = DG + offset_bh * $(NUM_BLOCK) * NUM_K * NUM_V +
      offset_d * NUM_V + offset_s + ($(NUM_BLOCK) - $(2)) * NUM_K * NUM_V
    D = D + offset_bh * $(NUM_BLOCK) + ($(NUM_BLOCK) - $(1))
    Dacc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
    for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
      S_i = tl.load(S)
      DS_i = tl.load(DS)
      d_i = tl.load(D)
      Dacc = DS_i + Dacc * d_i
      DG_i = tl.sum(Dacc * (S_i).to(tl.float32))
      tl.store(DG, (DG_i).to(tl.float32))
      tl.store(DI, (Dacc).to(tl.float32))
      S -= $(D_MODEL_K) * $(D_MODEL_V)
      DI -= $(D_MODEL_K) * $(D_MODEL_V)
      DS -= $(D_MODEL_K) * $(D_MODEL_V)
      DG -= NUM_K * NUM_V
      D -= $(1)
    }
    DL = DL + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :]
    DS_i = tl.load(DS)
    d_i = tl.load(D)
    Dacc = DS_i + Dacc * d_i
    tl.store(DL, (Dacc).to(DTYPE))
  }

/-- Source-only organization preserves the complete previous selected backward program. -/
theorem backward_previous (S D DI DG DL DS : RegionName)
    (H N K V BK BV : Nat) :
    backward S D DI DG DL DS H N K V BK BV =
      Old.backward .fp32 .fp32 .fp32 S D DI DG DL DS H N K V BK BV := rfl

theorem backward_candidate_previous (S D DI DG DL DS : RegionName)
    (H N K V BK BV : Nat) :
    backwardSumLeft S D DI DG DL DS H N K V BK BV =
      VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource.backwardSumLeft
        .fp32 .fp32 .fp32 S D DI DG DL DS H N K V BK BV := rfl
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
