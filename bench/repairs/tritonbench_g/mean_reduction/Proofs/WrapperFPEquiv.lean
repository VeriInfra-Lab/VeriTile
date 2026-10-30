import bench.repairs.tritonbench_g.mean_reduction.Kernels
import bench.repairs.tritonbench_g.mean_reduction.Proofs.Host

/-! Source FP32 instruction graph, including the N=1 lowering and the actual
winning lane of the dynamic butterfly tree. No associativity is assumed. -/
namespace VeriTile.Bench.Repairs.MeanReduction.FPEquiv
open VeriTile.Bench.TritonBenchG.MeanReduction Whole
noncomputable section

/-- Remaining native numerical sites. Their operands are the exact reached
32-bit operands in WholeLowered, not an entire reduction or output Tensor. -/
inductive NumericalSite where
  | addRNFP32 | signed32ToFP32RN | divFullFP32
  deriving DecidableEq, Repr

def numericalSites : List NumericalSite := [.addRNFP32,.signed32ToFP32RN,.divFullFP32]

specification mean_dim_repaired_fp {x : Tensor rank} {dims : List (Fin rank)}
    {nodup : dims.Nodup} {entry : Nat → BitVec 32} {preexisting : Nat → Prop}
    (binding : TensorBinding metadata x)
    (h : Admitted metadata x.shape axes dtype dims)
    (B : WholeLowered.Scalars) (winner : Nat → Fin 8)
    (rt : Runtime (BitVec 32) x dims nodup entry preexisting) (keepdim : Bool)
    (storage : ∀ c a, rt.compress=some c → rt.allocate c=some a → StorageBounds a) :
    metadata.shape=List.ofFn x.shape ∧
    (∃ result, runNative metadata axes dtype B winner rt keepdim=.ok result ∧
      HostOutcomes (fun row=>nativeFormula B winner x dims nodup entry h.2.positive row.val)
        rt keepdim result) ∧
    ∀ c a, rt.compress=some c → rt.allocate c=some a →
      ∀ pid t lane col, pid<Whole.gridSize (rows x.shape dims) →
        t<Whole.gridSize (columns x.shape dims) → lane<8 → col<8 →
        pid*8+lane<rows x.shape dims → t*8+col<columns x.shape dims →
        (nativeInputPointer c.base pid lane (columns x.shape dims) (t*8) col).toNat=
          4*(c.base+(pid*8+lane)*columns x.shape dims+(t*8+col)) ∧
        (nativeOutputPointer a.base pid lane).toNat=4*(a.base+pid*8+lane) := by
  exact ⟨binding.shape,native_outcomes h B winner rt keepdim,fun c a hc ha=>
    active_pointers a h.2 (storage c a hc ha)⟩

#axiomsClean mean_dim_repaired_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.FPEquiv
