import bench.repairs.tritonbench_g.quantize_global.Proofs.Execution
/-! Actual source equivalence between the two retained autotune block sizes.
Every invocation uses the same FP32 div.rn/multiply/clamp/llrint/INT8 graph.
Both complete kernel grids succeed and agree on all typed memory, including
the exterior frame. This needs no floating transformation law or primitive
conformance experiment. The original defective reciprocal graph is separate. -/
namespace QuantizeGlobalRepair.KernelNativeResults
open VeriTile VeriTile.Triton FP.Structural
noncomputable section

/-- Any nonempty sequences of the existing autotune configurations compute
the same source outputs. Scalar primitives include the actual signed64
llrint word and modular INT8 store; none are assumed equal to another model. -/
theorem configurations_equivalent {α : Type} [Inhabited α]
    (rest : Algebra α) (a : Scalar α) (p : Parameters) (sep : p.Separated)
    (xs : Nat → α) (maximum : α) (initial : State α) (inputs : Inputs p xs maximum initial)
    (left right : Nat → List Config)
    (leftNonempty : ∀ k, k < segmentCount p.n → left k ≠ [])
    (rightNonempty : ∀ k, k < segmentCount p.n → right k ≠ []) :
    ∃ l r,
      execute rest a p (schedule p.n left) initial = some l ∧
      execute rest a p (schedule p.n right) initial = some r ∧
      l.mem = r.mem ∧
      (∀ i, i < p.n → l.mem p.Out i = .mk .int (scalar a (xs i) maximum).toInt) ∧
      (∀ region i, (region ≠ p.Out ∨ p.n ≤ i) → l.mem region i = initial.mem region i) := by
  obtain ⟨l, hl, ml⟩ := whole_grid rest a p xs maximum left leftNonempty sep initial inputs
  obtain ⟨r, hr, mr⟩ := whole_grid rest a p xs maximum right rightNonempty sep initial inputs
  refine ⟨l, r, hl, hr, ml.trans mr.symm, ?_, ?_⟩
  · intro i hi
    rw [ml]
    simp [complete, hi]
  · intro region i outside
    rw [ml]
    rcases outside with region | index
    · simp [complete, region]
    · simp [complete, Nat.not_lt.mpr index]

/-- The two actual production block sizes, with identical input/absmax/output
ports and no numerical assumptions. -/
theorem block_sizes_equivalent {α : Type} [Inhabited α]
    (rest : Algebra α) (a : Scalar α) (p : Parameters) (sep : p.Separated)
    (xs : Nat → α) (maximum : α) (initial : State α) (inputs : Inputs p xs maximum initial) :
    ∃ l r,
      execute rest a p (schedule p.n (fun _ => [.block1024])) initial = some l ∧
      execute rest a p (schedule p.n (fun _ => [.block2048])) initial = some r ∧
      l.mem = r.mem ∧
      (∀ i, i < p.n → l.mem p.Out i = .mk .int (scalar a (xs i) maximum).toInt) ∧
      (∀ region i, (region ≠ p.Out ∨ p.n ≤ i) → l.mem region i = initial.mem region i) :=
  configurations_equivalent rest a p sep xs maximum initial inputs
    (fun _ => [.block1024]) (fun _ => [.block2048]) (by simp) (by simp)

#axiomsClean configurations_equivalent
#axiomsClean block_sizes_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.KernelNativeResults
