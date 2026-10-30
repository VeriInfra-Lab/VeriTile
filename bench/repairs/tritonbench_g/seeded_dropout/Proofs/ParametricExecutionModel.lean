import bench.repairs.tritonbench_g.seeded_dropout.Kernels
import bench.tritonbench_g.seeded_dropout.Proofs.RandomDomain
import VeriTile.Meta.StatementAudit

/-! Parametric execution model for proof induction, not the source authority.
The three `actual_*` equations below identify the complete actual pasted
program with this model at its selected block size. No statement is removed. -/
namespace VeriTile.Bench.Repairs.SeededDropout.ParametricExecutionModel
open VeriTile.Triton
noncomputable section

def sourceI32 (x_ptr output_ptr : RegionName) (n_elements B : Nat)
    (p : ℝ) (seed : Int) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = pid * $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB)
    offsets = block_start + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB))
    mask = offsets < $((n_elements : Nat))
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand($((_root_.VeriTile.Triton.Random.sourceIntCallValue VeriTile.Triton.Random.IntegerType.i32 seed : Int)), offsets)
    x_keep = random > $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))
    output = tl.where(x_keep, x / (1 - $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

theorem actual_i32 (X O : RegionName) (n : Nat) (p : ℝ) (seed : Int) :
    SeededDropout.sourceI32 X O n p seed =
      sourceI32 X O n 1024 p seed
        (TritonBenchG.SeededDropout.RandomDomain.arange_domain
          TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  simp only [SeededDropout.sourceI32,sourceI32,DSL.TypedStores.fp32CallValue]
  rfl

def sourceI64 (x_ptr output_ptr : RegionName) (n_elements B : Nat)
    (p : ℝ) (seed : Int) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = pid * $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB)
    offsets = block_start + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB))
    mask = offsets < $((n_elements : Nat))
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand($((_root_.VeriTile.Triton.Random.sourceIntCallValue VeriTile.Triton.Random.IntegerType.i64 seed : Int)), offsets)
    x_keep = random > $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))
    output = tl.where(x_keep, x / (1 - $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

theorem actual_i64 (X O : RegionName) (n : Nat) (p : ℝ) (seed : Int) :
    SeededDropout.sourceI64 X O n p seed =
      sourceI64 X O n 1024 p seed
        (TritonBenchG.SeededDropout.RandomDomain.arange_domain
          TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  simp only [SeededDropout.sourceI64,sourceI64,DSL.TypedStores.fp32CallValue]
  rfl

def sourceU64 (x_ptr output_ptr : RegionName) (n_elements B : Nat)
    (p : ℝ) (seed : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = pid * $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB)
    offsets = block_start + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB))
    mask = offsets < $((n_elements : Nat))
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand($((_root_.VeriTile.Triton.Random.sourceNatCallValue VeriTile.Triton.Random.IntegerType.u64 seed : Nat)), offsets)
    x_keep = random > $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))
    output = tl.where(x_keep, x / (1 - $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

theorem actual_u64 (X O : RegionName) (n : Nat) (p : ℝ) (seed : Nat) :
    SeededDropout.sourceU64 X O n p seed =
      sourceU64 X O n 1024 p seed
        (TritonBenchG.SeededDropout.RandomDomain.arange_domain
          TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  simp only [SeededDropout.sourceU64,sourceU64,DSL.TypedStores.fp32CallValue]
  rfl

def candidateI32 (x_ptr output_ptr : RegionName) (n_elements B : Nat)
    (p : ℝ) (seed : Int) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB) * pid
    offsets = block_start + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB))
    mask = offsets < $((n_elements : Nat))
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand($((_root_.VeriTile.Triton.Random.sourceIntCallValue VeriTile.Triton.Random.IntegerType.i32 seed : Int)), offsets)
    x_keep = random > $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))
    output = tl.where(x_keep, x / (1 - $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

theorem actual_candidate_i32 (X O : RegionName) (n : Nat) (p : ℝ) (seed : Int) :
    SeededDropout.candidateI32 X O n p seed =
      candidateI32 X O n 1024 p seed
        (TritonBenchG.SeededDropout.RandomDomain.arange_domain
          TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  simp only [SeededDropout.candidateI32,candidateI32,DSL.TypedStores.fp32CallValue]
  rfl

def candidateI64 (x_ptr output_ptr : RegionName) (n_elements B : Nat)
    (p : ℝ) (seed : Int) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB) * pid
    offsets = block_start + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB))
    mask = offsets < $((n_elements : Nat))
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand($((_root_.VeriTile.Triton.Random.sourceIntCallValue VeriTile.Triton.Random.IntegerType.i64 seed : Int)), offsets)
    x_keep = random > $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))
    output = tl.where(x_keep, x / (1 - $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

theorem actual_candidate_i64 (X O : RegionName) (n : Nat) (p : ℝ) (seed : Int) :
    SeededDropout.candidateI64 X O n p seed =
      candidateI64 X O n 1024 p seed
        (TritonBenchG.SeededDropout.RandomDomain.arange_domain
          TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  simp only [SeededDropout.candidateI64,candidateI64,DSL.TypedStores.fp32CallValue]
  rfl

def candidateU64 (x_ptr output_ptr : RegionName) (n_elements B : Nat)
    (p : ℝ) (seed : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.region x_ptr = tl.float32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB) * pid
    offsets = block_start + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB))
    mask = offsets < $((n_elements : Nat))
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand($((_root_.VeriTile.Triton.Random.sourceNatCallValue VeriTile.Triton.Random.IntegerType.u64 seed : Nat)), offsets)
    x_keep = random > $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))
    output = tl.where(x_keep, x / (1 - $((VeriTile.Triton.DSL.TypedStores.fp32CallValue p : ℝ))), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

theorem actual_candidate_u64 (X O : RegionName) (n : Nat) (p : ℝ) (seed : Nat) :
    SeededDropout.candidateU64 X O n p seed =
      candidateU64 X O n 1024 p seed
        (TritonBenchG.SeededDropout.RandomDomain.arange_domain
          TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  simp only [SeededDropout.candidateU64,candidateU64,DSL.TypedStores.fp32CallValue]
  rfl

def source (kind : SeedKind) (X O : RegionName) (n B : Nat) (p : ℝ)
    (seed : SeedCarrier kind) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  match kind,seed with
  | .i32,s => sourceI32 X O n B p s hB
  | .i64,s => sourceI64 X O n B p s hB
  | .u64,s => sourceU64 X O n B p s hB

theorem actual (kind : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ)
    (seed : SeedCarrier kind) :
    kernel kind X O n p seed = source kind X O n 1024 p seed
      (TritonBenchG.SeededDropout.RandomDomain.arange_domain
        TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  cases kind
  · exact actual_i32 X O n p seed
  · exact actual_i64 X O n p seed
  · exact actual_u64 X O n p seed


def candidate (kind : SeedKind) (X O : RegionName) (n B : Nat) (p : ℝ)
    (seed : SeedCarrier kind) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  match kind,seed with
  | .i32,s => candidateI32 X O n B p s hB
  | .i64,s => candidateI64 X O n B p s hB
  | .u64,s => candidateU64 X O n B p s hB

theorem actual_candidate (kind : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ)
    (seed : SeedCarrier kind) :
    indexCommutedKernel kind X O n p seed = candidate kind X O n 1024 p seed
      (TritonBenchG.SeededDropout.RandomDomain.arange_domain
        TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal) := by
  cases kind
  · exact actual_candidate_i32 X O n p seed
  · exact actual_candidate_i64 X O n p seed
  · exact actual_candidate_u64 X O n p seed

#axiomsClean actual_i32
#axiomsClean actual_i64
#axiomsClean actual_u64
#axiomsClean actual_candidate_i32
#axiomsClean actual_candidate_i64
#axiomsClean actual_candidate_u64
#axiomsClean actual
#axiomsClean actual_candidate
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.SeededDropout.ParametricExecutionModel
