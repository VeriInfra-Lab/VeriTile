import bench.tritonbench_g.chunk_gate_recurrence.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.GuardedOptionalRebinding
import VeriTile.Meta.StatementAudit

/-! Complete repaired recurrence bodies and the existing sum-left comparison.
Only invocation metadata is outside the copied Python bodies. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def forwardSignature : CallSignature where
  pointers := [("S",.fp32),("d",.fp32),("O",.fp32),("last_kv",.fp32)]
  scalars := [("NUM_HEAD",.integer),("NUM_BLOCK",.integer)]
  constexpr := ["D_MODEL_K","D_MODEL_V","BLOCK_MODEL_K","BLOCK_MODEL_V","SEED_B","SEED_H","SEED_K","SEED_V"]
  readPorts := ["S","d","last_kv"]
  writePorts := ["O"]

def backwardSignature (source : FloatDType) (gate incoming : Storage) : CallSignature where
  pointers := [("d", match gate with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),
    ("DI", match incoming with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),
    ("DG", match gate with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),
    ("DS", match incoming with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16)]
  floatingPointers := [("S",source),("DL",source)]
  scalars := [("NUM_HEAD",.integer),("NUM_BLOCK",.integer)]
  constexpr := ["D_MODEL_K","D_MODEL_V","BLOCK_MODEL_K","BLOCK_MODEL_V"]
  readPorts := ["S","d","DS"]
  writePorts := ["DI","DG","DL"]

def forwardKernel (S d O last_kv : RegionName) (NUM_HEAD NUM_BLOCK : Nat)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V SEED_B SEED_H SEED_K SEED_V : Nat) (hasSeed : Bool) : ComputeKernel :=
  with_triton_call forwardSignature call_args [NUM_HEAD := NUM_HEAD, NUM_BLOCK := NUM_BLOCK] guarded_optional_pointers [last_kv := hasSeed] in triton {
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    if last_kv is not None:
        last_kv = last_kv + (offset_bh // NUM_HEAD) * SEED_B + (offset_bh % NUM_HEAD) * SEED_H + (offset_d * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None]) * SEED_K + (offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]) * SEED_V
        acc = tl.load(last_kv).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V
    d = d + offset_bh * NUM_BLOCK
    for i in range(NUM_BLOCK-1):
        d_i = tl.load(d)
        S_i = tl.load(S)
        acc = acc * d_i + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        d += 1
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V
  }

def backwardKernel (source : FloatDType) (gate incoming : Storage) (S d DI DG DL DS : RegionName)
    (NUM_HEAD NUM_BLOCK : Nat) (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : ComputeKernel :=
  with_triton_call (backwardSignature source gate incoming) call_args [NUM_HEAD := NUM_HEAD, NUM_BLOCK := NUM_BLOCK] in triton {
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    # offset_h = offset_bh % NUM_HEAD
    NUM_K = D_MODEL_K // BLOCK_MODEL_K
    NUM_V = D_MODEL_V // BLOCK_MODEL_V
    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    DI = DI + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    DG = DG + offset_bh * NUM_BLOCK * NUM_K * NUM_V + offset_d * NUM_V + offset_s + (NUM_BLOCK - 2) * NUM_K * NUM_V

    d = d + offset_bh * NUM_BLOCK + (NUM_BLOCK - 1)

    Dacc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        d_i = tl.load(d)
        Dacc = Dacc * d_i + DS_i
        DG_i = tl.sum(Dacc * S_i.to(tl.float32))

        tl.store(DG, DG_i.to(DG.dtype.element_ty))
        tl.store(DI, Dacc.to(DI.dtype.element_ty))

        S -= D_MODEL_K * D_MODEL_V
        DI -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V
        DG -= NUM_K * NUM_V
        d -= 1

    DL = DL + offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    DS_i = tl.load(DS)
    d_i = tl.load(d)
    Dacc = Dacc * d_i + DS_i
    tl.store(DL, Dacc.to(DL.dtype.element_ty))
  }

def forwardBodySumLeftKernel (S d O last_kv : RegionName) (NUM_HEAD NUM_BLOCK : Nat)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V SEED_B SEED_H SEED_K SEED_V : Nat) (hasSeed : Bool) : ComputeKernel :=
  with_triton_call forwardSignature call_args [NUM_HEAD := NUM_HEAD, NUM_BLOCK := NUM_BLOCK] guarded_optional_pointers [last_kv := hasSeed] in triton {
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    if last_kv is not None:
        last_kv = last_kv + (offset_bh // NUM_HEAD) * SEED_B + (offset_bh % NUM_HEAD) * SEED_H + (offset_d * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None]) * SEED_K + (offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]) * SEED_V
        acc = tl.load(last_kv).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V
    d = d + offset_bh * NUM_BLOCK
    for i in range(NUM_BLOCK-1):
        d_i = tl.load(d)
        S_i = tl.load(S)
        acc = S_i + acc * d_i
        tl.store(O, acc.to(O.dtype.element_ty))
        d += 1
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V
  }

def backwardBodySumLeftKernel (source : FloatDType) (gate incoming : Storage) (S d DI DG DL DS : RegionName)
    (NUM_HEAD NUM_BLOCK : Nat) (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : ComputeKernel :=
  with_triton_call (backwardSignature source gate incoming) call_args [NUM_HEAD := NUM_HEAD, NUM_BLOCK := NUM_BLOCK] in triton {
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)

    # offset_h = offset_bh % NUM_HEAD
    NUM_K = D_MODEL_K // BLOCK_MODEL_K
    NUM_V = D_MODEL_V // BLOCK_MODEL_V
    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    DI = DI + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    DG = DG + offset_bh * NUM_BLOCK * NUM_K * NUM_V + offset_d * NUM_V + offset_s + (NUM_BLOCK - 2) * NUM_K * NUM_V

    d = d + offset_bh * NUM_BLOCK + (NUM_BLOCK - 1)

    Dacc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        d_i = tl.load(d)
        Dacc = DS_i + Dacc * d_i
        DG_i = tl.sum(Dacc * S_i.to(tl.float32))

        tl.store(DG, DG_i.to(DG.dtype.element_ty))
        tl.store(DI, Dacc.to(DI.dtype.element_ty))

        S -= D_MODEL_K * D_MODEL_V
        DI -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V
        DG -= NUM_K * NUM_V
        d -= 1

    DL = DL + offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    DS_i = tl.load(DS)
    d_i = tl.load(d)
    Dacc = DS_i + Dacc * d_i
    tl.store(DL, Dacc.to(DL.dtype.element_ty))
  }

/-- Fixed 64×16 FP32 invocation retained by the full program proofs. -/
def forward (S D O seed : RegionName) (H N K V SB SH SK SV : Nat) (hasSeed : Bool) :=
  forwardKernel S D O seed H N K V 64 16 SB SH SK SV hasSeed

def forwardSumLeft (S D O seed : RegionName) (H N K V SB SH SK SV : Nat) (hasSeed : Bool) :=
  forwardBodySumLeftKernel S D O seed H N K V 64 16 SB SH SK SV hasSeed

def backwardFamily (dtype : FloatDType) (S D DI DG DL DS : RegionName)
    (NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :=
  backwardKernel dtype .fp32 .fp32 S D DI DG DL DS NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V

def backwardSumLeftFamily (dtype : FloatDType) (S D DI DG DL DS : RegionName)
    (NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :=
  backwardBodySumLeftKernel dtype .fp32 .fp32 S D DI DG DL DS NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V

abbrev backward := backwardFamily .fp32
abbrev backwardSumLeft := backwardSumLeftFamily .fp32
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels
