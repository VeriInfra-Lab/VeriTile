import bench.tritonbench_g.rmsnorm_triton.Proofs.Contracts

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.FiniteWitness
open VeriTile.Triton
noncomputable section

/-- Finite, disjoint input, weight and output allocations, with padding. -/
def allocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("X", 21), ("W", 7), ("Y", 22)]

def initial : BlockState where
  mem := fun r _ => if r = "X" then .real 2
    else if r = "W" then .real 3 else .of .int 37
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2

def io (v : Bool) : StreamEmitMasked2DKernelIO₂ :=
  typedIO .fp16 .fp16 .fp16 v "X" "W" "Y" 11 5 2 3 11 5 2 3 2 5

private theorem formula_two (t : Fin (rmsNumSteps 3 2)) (j : Fin 2) :
    rmsStreamSpec 3 2 5 (fun _ _ => 2) (fun _ _ => 3) t j = 2 := by
  have hsum : rmsStreamSumSq 3 2 (fun _ _ => 2) = 12 := by
    change (∑ u : Fin 2, ∑ e : Fin 2,
      if u.val * 2 + e.val < 3 then (2 : ℝ) ^ 2 else 0) = 12
    norm_num [Fin.sum_univ_succ]
  norm_num [rmsStreamSpec, hsum]

/-- An inhabited fp16 storage profile: both program IDs are nonzero, both
loops run twice, the last tile is masked, and all column strides are nonunit.
The contract includes finite access safety, successful flat execution, nonzero
outputs and preservation of every other cell, including typed sentinels. -/
theorem execution (v : Bool) :
    (io v).kernel.toAlgKernel.TraceSafeR .triv allocation.extent initial ∧
    ∃ t, exec (allocation.flattenKernel (io v).kernel.toAlgKernel)
      (allocation.flattenState initial) = some t ∧
      (∀ (u : Fin (rmsNumSteps 3 2)) (j : Fin 2), u.val * 2 + j.val < 3 →
        t.readMem allocation.flat
          (allocation.addr "Y" (16 + (u.val * 2 + j.val) * 2)) = 2) ∧
      (∀ r off, (r ≠ allocation.flat ∨
          ∀ (u : Fin (rmsNumSteps 3 2)) (j : Fin 2), u.val * 2 + j.val < 3 →
            off ≠ allocation.addr "Y" (16 + (u.val * 2 + j.val) * 2)) →
        t.mem r off = (allocation.flattenState initial).mem r off) := by
  have h := typed_exact_io .fp16 .fp16 .fp16 v "X" "W" "Y"
    11 5 2 3 11 5 2 3 2 5 (by decide) (by decide) (by decide) (by decide)
    allocation (FlatAlloc.ofList_disjoint _ _ (by decide)) rfl
    (FlatAlloc.ofList_closed _ _) 1 1 (fun _ _ => 2) (fun _ _ => 3) initial
    rfl rfl rfl
  have hx : ∀ (t : Fin (rmsNumSteps 3 2)) (j : Fin 2), t.val * 2 + j.val < 3 →
      1 * 11 + 1 * 5 + (t.val * 2 + j.val) * 2 < allocation.extent "X" := by
    intro t j hj
    change 16 + (t.val * 2 + j.val) * 2 < 21
    omega
  have hw : ∀ (t : Fin (rmsNumSteps 3 2)) (j : Fin 2), t.val * 2 + j.val < 3 →
      (t.val * 2 + j.val) * 3 < allocation.extent "W" := by
    intro t j hj
    change (t.val * 2 + j.val) * 3 < 7
    omega
  have ho : ∀ (t : Fin (rmsNumSteps 3 2)) (j : Fin 2), t.val * 2 + j.val < 3 →
      1 * 11 + 1 * 5 + (t.val * 2 + j.val) * 2 < allocation.extent "Y" := by
    intro t j hj
    change 16 + (t.val * 2 + j.val) * 2 < 22
    omega
  obtain ⟨hsafe, t, he, hv, hf⟩ := h hx hw ho
    (by intro u j hj; simp [typedIO, rmsnormKernelIO, initial, BlockState.readMem])
    (by intro u j hj; simp [typedIO, rmsnormKernelIO, initial, BlockState.readMem])
  refine ⟨hsafe, t, he, ?_, ?_⟩
  · intro u j hj
    simpa [typedIO, rmsnormKernelIO, formula_two] using hv u j hj
  · simpa [typedIO, rmsnormKernelIO] using hf

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormTriton.FiniteWitness
