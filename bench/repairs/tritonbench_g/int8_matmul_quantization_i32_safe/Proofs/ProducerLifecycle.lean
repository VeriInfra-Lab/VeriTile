import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Admission

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerLifecycle
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
variable {F : Type} [Inhabited F]

/-- Each benchmark records one of the four retained producer configurations.
Algebra fields remain scalar instructions/reduction structure. -/
structure Invocation (F : Type) (p : KernelProducerGrid.Geometry) where
  configIndex : Fin 4
  algebra : Algebra F
  integer8 : F → Byte
  schedule : Producer.Schedule p

def trials (p : KernelProducerGrid.Geometry) : List (Invocation F p) → State F → Option (State F)
  | [],state => some state
  | v::vs,state => do
      let next ← KernelProducerGrid.run v.algebra v.integer8 p v.schedule.order state
      trials p vs next

/-- Producer autotuning cannot change the input slice or other allocations.
Consequently final native/Real producer targets use the original input,
regardless of the number/order of benchmark invocations. -/
theorem complete (p : KernelProducerGrid.Geometry) (vs : List (Invocation F p)) (state : State F) :
    ∃ final,trials p vs state=some final ∧
      (∀ (region : RegionName) i, (region ≠ "q" ∨ p.rows*p.columns ≤ i) →
        (region ≠ "scales" ∨ p.rows ≤ i) → final.mem region i=state.mem region i) := by
  induction vs generalizing state with
  | nil => exact ⟨state,rfl,fun _ _ _ _ => rfl⟩
  | cons v vs ih =>
    obtain ⟨next,he,_,_,hf⟩:=KernelProducerGrid.source_correct v.algebra v.integer8 p v.schedule state
    obtain ⟨final,ht,frame⟩:=ih next
    refine ⟨final,by simp [trials,he,ht],?_⟩
    intro region i hq hs
    exact (frame region i hq hs).trans (hf region i hq hs)

theorem input_reads (p : KernelProducerGrid.Geometry) (values : Nat → Nat → F) (vs : List (Invocation F p))
    (state : State F) (reads : KernelProducerTargets.Reads p values state) :
    ∃ final,trials p vs state=some final ∧ KernelProducerTargets.Reads p values final := by
  obtain ⟨final,he,hf⟩:=complete p vs state
  refine ⟨final,he,?_⟩
  intro r k hr hk
  rw [hf "x" _ (Or.inl (by decide)) (Or.inl (by decide))]
  exact reads r k hr hk

#axiomsClean complete
#axiomsClean input_reads
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerLifecycle
