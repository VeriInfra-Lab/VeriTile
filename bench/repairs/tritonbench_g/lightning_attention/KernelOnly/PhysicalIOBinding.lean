import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.InputBinding
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.IOResults
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.IOSafety
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.IOBounds
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.TensorValues
import bench.repairs.tritonbench_g.chunk_delta_fwd.Readback
/-! Concrete original Tensor readers bind the same logical inputs and source
memory as the public IO. This bridge retains original storage offsets/read
aliases and the incoming contiguous preparation equality. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PhysicalIO
open VeriTile Triton FP.Structural
noncomputable section

def forwardInputs (r : Request .forward) (read : Nat → ℝ) : (forwardIO r.dimensions).Inputs :=
  fun port => match port with
  | ⟨0,_⟩ => fun (h,t,a,()) => tensorQ r read h t a
  | ⟨1,_⟩ => fun (h,t,a,()) => tensorK r read h t a
  | ⟨2,_⟩ => fun (h,t,c,()) => tensorV r read h t c
  | ⟨3,_⟩ => fun _ => (0 : ℝ)

def backwardInputs (r : Request .backward) (read : Nat → ℝ) : (backwardIO r.dimensions).Inputs :=
  fun port => match port with
  | ⟨0,_⟩ => fun (h,t,a,()) => tensorQ r read h t a
  | ⟨1,_⟩ => fun (h,t,a,()) => tensorK r read h t a
  | ⟨2,_⟩ => fun (h,t,c,()) => tensorV r read h t c
  | ⟨3,_⟩ => fun (h,t,c,()) => tensorG r read h t c

theorem forward_binding (r : Request .forward) (h : Contract r) (read : Nat → ℝ) :
    (forwardIO r.dimensions).InputBinding (forwardInputs r read)
      (ChunkDeltaFwd.memoryState (initialReal r read)) := by
  intro ⟨port,hport⟩ j active
  have hp : port < 4 := hport
  interval_cases port
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    exact congrFun (congrFun (input_Q r h read head) time) feature
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    exact congrFun (congrFun (input_K r h read head) time) feature
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    exact congrFun (congrFun (input_V r h read head) time) feature
  · have : False := by simpa [forwardIO] using active
    exact this.elim

theorem backward_binding (r : Request .backward) (h : Contract r) (read : Nat → ℝ)
    (prepared : Prepared r read) :
    (backwardIO r.dimensions).InputBinding (backwardInputs r read)
      (ChunkDeltaFwd.memoryState (initialReal r read)) := by
  intro ⟨port,hport⟩ j active
  have hp : port < 4 := hport
  interval_cases port
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    exact congrFun (congrFun (input_Q r h read head) time) feature
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    exact congrFun (congrFun (input_K r h read head) time) feature
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    exact congrFun (congrFun (input_V r h read head) time) feature
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    exact congrFun (congrFun (input_G r h read prepared head) time) feature

theorem forward_formula (r : Request .forward) (read : Nat → ℝ)
    (head : Fin (r.dimensions.B*r.dimensions.H)) (t : Fin r.dimensions.N) (c : Fin r.dimensions.E) :
    forwardFormula r.dimensions (forwardInputs r read) (0 : Fin 1) (head,t,c,()) =
      Target.output (tensorQ r read head) (tensorK r read head) (tensorV r read head) t c := rfl

theorem backward_formula (r : Request .backward) (read : Nat → ℝ)
    (head : Fin (r.dimensions.B*r.dimensions.H)) :
    (∀ t a, backwardFormula r.dimensions (backwardInputs r read) (0 : Fin 3) (head,t,a,()) =
      Target.dq (tensorK r read head) (tensorV r read head) (tensorG r read head) t a) ∧
    (∀ t a, backwardFormula r.dimensions (backwardInputs r read) (1 : Fin 3) (head,t,a,()) =
      Target.dk (tensorQ r read head) (tensorV r read head) (tensorG r read head) t a) ∧
    (∀ t c, backwardFormula r.dimensions (backwardInputs r read) (2 : Fin 3) (head,t,c,()) =
      Target.dv (tensorQ r read head) (tensorK r read head) (tensorG r read head) t c) :=
  ⟨fun _ _ => rfl,fun _ _ => rfl,fun _ _ => rfl⟩

/-- The actual physical reader receives the same complete source schedule,
proved source access safety, values and exterior frame as the public IO. -/
theorem forward_application (r : Request .forward) (h : Contract r) (read : Nat → ℝ) :
    ScheduleSafe (IOProof.forwardCapacity r.dimensions)
      ((forwardIO r.dimensions).schedule (forwardInputs r read))
      (ChunkDeltaFwd.memoryState (initialReal r read)) ∧
    ∃ final, runSchedule ((forwardIO r.dimensions).schedule (forwardInputs r read))
        (ChunkDeltaFwd.memoryState (initialReal r read)) = some final ∧
      (forwardIO r.dimensions).OutputBinding (forwardInputs r read)
        (forwardFormula r.dimensions (forwardInputs r read)) final ∧
      (forwardIO r.dimensions).Frame (forwardInputs r read)
        (ChunkDeltaFwd.memoryState (initialReal r read)) final :=
  ⟨IOProof.forward_safety _ h.dimensions _ _ (IOProof.forward_in_bounds _ _) _,
   IOProof.forward_result _ h.dimensions _ _ (forward_binding r h read)⟩

theorem backward_application (r : Request .backward) (h : Contract r) (read : Nat → ℝ)
    (prepared : Prepared r read) :
    ScheduleSafe (IOProof.backwardCapacity r.dimensions)
      ((backwardIO r.dimensions).schedule (backwardInputs r read))
      (ChunkDeltaFwd.memoryState (initialReal r read)) ∧
    ∃ final, runSchedule ((backwardIO r.dimensions).schedule (backwardInputs r read))
        (ChunkDeltaFwd.memoryState (initialReal r read)) = some final ∧
      (backwardIO r.dimensions).OutputBinding (backwardInputs r read)
        (backwardFormula r.dimensions (backwardInputs r read)) final ∧
      (backwardIO r.dimensions).Frame (backwardInputs r read)
        (ChunkDeltaFwd.memoryState (initialReal r read)) final :=
  ⟨IOProof.backward_safety _ _ _ (IOProof.backward_in_bounds _ _) _,
   IOProof.backward_result _ h.dimensions _ _ (backward_binding r h read prepared)⟩

#axiomsClean forward_application
#axiomsClean backward_application

#axiomsClean forward_binding
#axiomsClean backward_binding
#axiomsClean forward_formula
#axiomsClean backward_formula
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PhysicalIO
