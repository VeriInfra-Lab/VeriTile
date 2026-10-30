import bench.tritonbench_g.chunk_delta_fwd.Proofs.PairContext
namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairSource
open VeriTile Triton FP.Structural FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem safe_ref (M : Algebra α) (p : Option ComputeDType) (bounds : RegionName → Nat)
    (s : State α) (d : TileDType) (sh : TileShape) (n : RegName) :
    safeOp M p bounds s (.ref d sh n) := by rw [safeOp_unfold]; trivial
private theorem safe_nat (M : Algebra α) (p : Option ComputeDType) (bounds : RegionName → Nat)
    (s : State α) (n : Nat) : safeOp M p bounds s (.constNat n) := by rw [safeOp_unfold]; trivial
private theorem safe_mul (M : Algebra α) (p : Option ComputeDType) (bounds : RegionName → Nat)
    (s : State α) (a b : Op .nat []) :
    safeOp M p bounds s (.mul .nat .nil a b) ↔ safeOp M p bounds s a ∧ safeOp M p bounds s b := by
  rw [safeOp_unfold]
private theorem safe_add (M : Algebra α) (p : Option ComputeDType) (bounds : RegionName → Nat)
    (s : State α) (a b : Op .nat []) :
    safeOp M p bounds s (.add .nat .nil a b) ↔ safeOp M p bounds s a ∧ safeOp M p bounds s b := by
  rw [safeOp_unfold]
private theorem safe_ptr (M : Algebra α) (p : Option ComputeDType) (bounds : RegionName → Nat)
    (s : State α) (r : RegionName) (base : Op .nat []) (parent block strides : List Nat)
    (offsets : List (Op .nat [])) :
    safeOp M p bounds s (.makeBlockPtrDynOffsets r base parent block strides offsets) ↔
      safeOp M p bounds s base ∧ (∀ e ∈ offsets, safeOp M p bounds s e) := by
  rw [safeOp_unfold]

/-- Complete source execution and trace safety are unchanged, including the
optional state branches and every history pointer in the outer range loop. -/
theorem complete (M : Algebra α) (p : Parameters) :
    CodeAgree M (kernel p).surfaceBody (addressFold p).surfaceBody := by
  unfold kernel ChunkDeltaFwd.kernel addressFold ComputeKernel.surfaceBody
  repeat' first
    | apply code_refl
    | apply cons
    | apply stmt_refl
    | apply branch
    | apply range_stmt
  all_goals
    constructor
    · intro s
      cases hb : s.regs .nat [] "i_bh" <;> cases ht : s.regs .nat [] "i_t" <;>
        simp [FP.Structural.step, evalExpr, evalComputeOp, evalOp_unfold, numeric,
        FP.Structural.bop, Nat.mul_assoc, hb, ht, Broadcast.leftIndex, Broadcast.rightIndex]
    · intro bounds s
      simp only [stepSafe_unfold, safeExpr_unfold]
      simp [safe_ptr, safe_mul, safe_add, safe_ref, safe_nat]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairSource
