import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Wrapper
open VeriTile.Triton FP.Structural
open Checked Physical
noncomputable section

/-- Pointwise saved-storage binding is derived from actual forward stores or
readonly input frames, rather than assumed as a saved-output equation. -/
theorem saved_binding {M α δ β : Type} [Inhabited α] (api : API M δ β) (ops : Ops α)
    (evaluate : α → δ) (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (BD : Nat) (positive : 0<BD)
    (sourceMemory : RegionName → Nat → α) (initial incoming : M)
    (loads : ∀ r a,evaluate (sourceMemory (name r) a)=api.read initial (key runtime ⟨r,a⟩))
    (saved : SavedStorage api runtime
      (commit api c U S V runtime Bool.true
        (fun r a => evaluate (Execution.read (Closed.first ops c BD U S sourceMemory) (name r) a)) initial) incoming)
    (r : Role) (held : r=.g ∨ r=.h0 ∨ r=.o ∨ r=.ht)
    (pr : present U S V r=Bool.true) (a : Nat) (bound : a<extent c r) :
    api.read incoming (key runtime ⟨r,a⟩)=
      evaluate (Execution.read (Closed.first ops c BD U S sourceMemory) (name r) a) := by
  rw [saved.read r held pr a bound]
  rcases held with rfl|rfl|rfl|rfl
  · rw [forward_keeps api c U S V runtime accepted _ initial ⟨.g,a⟩ ⟨pr,bound⟩ (Or.inr (Or.inl rfl))]
    have framed := (Closed.first_complete ops c BD positive U S sourceMemory).2.2.2 "g" a
      (by simp [Execution.ForwardRegion])
    change _=evaluate (((Closed.first ops c BD U S sourceMemory).mem "g" a).read .real)
    rw [framed]
    exact (loads .g a).symm
  · rw [forward_keeps api c U S V runtime accepted _ initial ⟨.h0,a⟩ ⟨pr,bound⟩ (Or.inr (Or.inr rfl))]
    have framed := (Closed.first_complete ops c BD positive U S sourceMemory).2.2.2 "h0" a
      (by simp [Execution.ForwardRegion])
    change _=evaluate (((Closed.first ops c BD U S sourceMemory).mem "h0" a).read .real)
    rw [framed]
    exact (loads .h0 a).symm
  · exact readback api c U S V runtime accepted Bool.true _ initial ⟨.o,a⟩
      ((cells_mem c S Bool.true _).mpr (Or.inl ⟨rfl,bound⟩))
  · exact readback api c U S V runtime accepted Bool.true _ initial ⟨.ht,a⟩
      ((cells_mem c S Bool.true _).mpr (Or.inr ⟨pr,rfl,bound⟩))

theorem mathematical_saved {M β : Type} (api : API M ℝ β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true) (BD : Nat) (positive : 0<BD)
    (initial incoming : M)
    (saved : SavedStorage api runtime (mathematicalForward api c U S V runtime BD initial).memory incoming)
    (r : Role) (held : r=.g ∨ r=.h0 ∨ r=.o ∨ r=.ht) (pr : present U S V r=Bool.true)
    (a : Nat) (bound : a<extent c r) :
    api.read incoming (key runtime ⟨r,a⟩)=
      Execution.read (Closed.first realOps c BD U S (memory api 0 runtime initial)) (name r) a := by
  apply saved_binding api realOps id c U S V runtime accepted BD positive _ initial incoming _ saved r held pr a bound
  intro r a
  simp [memory,role_name]

theorem native_saved {M β : Type} (api : API M (BitVec 32) β) (p : Native.Instructions)
    (schedule : Native.Schedule) (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (BD : Nat) (positive : 0<BD) (initial incoming : M)
    (saved : SavedStorage api runtime (nativeForward api p schedule c U S V runtime BD initial).memory incoming)
    (r : Role) (held : r=.g ∨ r=.h0 ∨ r=.o ∨ r=.ht) (pr : present U S V r=Bool.true)
    (a : Nat) (bound : a<extent c r) :
    api.read incoming (key runtime ⟨r,a⟩)=
      Native.observe p schedule (Closed.first Native.sourceOps c BD U S (terms api runtime initial)) (name r) a := by
  apply saved_binding api Native.sourceOps (Native.eval p schedule) c U S V runtime accepted BD positive
    _ initial incoming _ saved r held pr a bound
  intro r a
  simp [terms,Native.eval,memory,role_name]

/-- The one new zero allocation implements an absent output cotangent. This
is a local zeros_like law, independent of recurrence or gradient semantics. -/
def MissingZero {M α β : Type} (api : API M α β) (zero : α) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (m : M) : Prop :=
  hasDo=Bool.false → ∀ a,a<c.rows → api.read m (key runtime ⟨.incoming,a⟩)=zero

#axiomsClean mathematical_saved
#axiomsClean native_saved
#specNonCircular MissingZero avoiding [Execution.gridRun,Closed.first,Closed.second,state,dx,dg]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Wrapper
