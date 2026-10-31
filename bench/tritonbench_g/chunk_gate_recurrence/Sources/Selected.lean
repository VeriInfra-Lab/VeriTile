import VeriTile.Triton.DSL
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
open VeriTile.Triton
/-- Complete unchanged FP32 backward body, including DG and final DL. -/
def backwardFamily (dtype : FloatDType) (S D DI DG DL DS : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :=
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
def backwardSumLeftFamily (dtype : FloatDType) (S D DI DG DL DS : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :=
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

end VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
