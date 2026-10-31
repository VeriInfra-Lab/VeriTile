import bench.tritonbench_g.chunk_gate_recurrence.Sources.Mathematical_Kernels
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
open VeriTile.Triton

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

abbrev backward := backwardFamily .fp32
abbrev backwardSumLeft := backwardSumLeftFamily .fp32
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
