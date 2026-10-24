import bench.repairs.tritonbench_g.uniform_sampling.Proofs.KernelExecution
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.UniformSampling.KernelEquivalence
open VeriTile.Triton FP.Structural KernelExecution
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def original (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  kernel y N seed offset lower upper B hB

def indexCommuted (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  indexCommutedKernel y N seed offset lower upper B hB

/-- Unconditional equality of the actual source executions, including failure.
No arithmetic law about the opaque floating carrier is a premise. -/
theorem index_commuted_equiv {α : Type} [Inhabited α] (M : Algebra α)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (s : State α) :
    FP.Structural.exec M (original y N seed offset lower upper B hB) s =
      FP.Structural.exec M (indexCommuted y N seed offset lower upper B hB) s :=
  KernelSource.execution_eq M y N seed offset lower upper B hB s

/-- Both displayed complete implementations succeed and agree on every typed
memory cell. FMA availability and seed/index ABI premises are structural model
conditions. Bounds remain arbitrary FP32 words, including special values. -/
theorem successful_index_commutation {α : Type} [Inhabited α]
    (M : Algebra α) (fused : α → α → α → α) (primitive : M.fma .fp32 = some fused)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat) (s : State α)
    (_nonempty : 0<N) (_extent : N≤2^34)
    (_block : B=512 ∨ B=1024)
    (_launched : pid < TritonBenchG.UniformSampling.SourceDomain.grid N B)
    (seedABI : Random.IntegerType.i64.inRange seed = Bool.true)
    (_offsetABI : Random.IntegerType.i64.inRange offset = Bool.true) :
    ∃ a b,
      FP.Structural.exec M (kernel y N seed offset lower upper B hB) (entry pid s) = some a ∧
      FP.Structural.exec M (indexCommutedKernel y N seed offset lower upper B hB) (entry pid s) = some b ∧
      a.mem = b.mem ∧
      (∀ r i,KernelStores.Outside y N pid B r i → a.mem r i = s.mem r i) := by
  refine ⟨result M fused y N seed offset lower upper B (entry pid s),
    result M fused y N seed offset lower upper B (entry pid s),?_,?_,rfl,?_⟩
  · exact executes M fused primitive Bool.false y N seed offset lower upper B hB (entry pid s) seedABI
  · exact executes M fused primitive Bool.true y N seed offset lower upper B hB (entry pid s) seedABI
  · exact fun r i outside => frame M fused y N seed offset lower upper B (entry pid s) r i outside

theorem original_source (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) :
    original "out_ptr" N seed offset lower upper B hB = originalKernel N seed offset lower upper B hB := rfl

theorem candidate_source (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) :
    indexCommuted "out_ptr" N seed offset lower upper B hB = candidateKernel N seed offset lower upper B hB := rfl

#axiomsClean index_commuted_equiv
#axiomsClean successful_index_commutation
#axiomsClean original_source
#axiomsClean candidate_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelEquivalence
