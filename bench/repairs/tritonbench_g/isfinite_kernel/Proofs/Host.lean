import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.SourceBridge
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Storage

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Checked
open VeriTile.Triton
noncomputable section

structure Operand where
  identity : Nat
  metadata : ViewMetadata

structure Result (M : Type) where
  output : Operand
  memory : M
  currentDevice : Nat

inductive Error where
  | input (reason : InputError)
  | launch

variable {M A : Type} {width : Nat} (ops : Memory.Primitives M A width)

/-- A successful runtime executes the direct chunks. A CUDA launch error is
an exception, not a mathematical output. Empty input bypasses device setup
and launch and returns the exact caller-supplied output object. -/
def run (classify : A → Bool) (x y : Operand) (before : M) (savedDevice : Nat)
    (launchesSucceed : Bool) : Except Error (Result M) := do
  match checkInputs (.view x.metadata) (.view y.metadata) with
  | .error e => .error (.input e)
  | .ok () =>
    if x.metadata.n=0 then .ok ⟨y,before,savedDevice⟩
    else if launchesSucceed then .ok ⟨y,
      Lowering.allChunks ops classify x.metadata.view y.metadata.outputView x.metadata.n
        (List.range (chunkCount x.metadata.n)) before,savedDevice⟩
    else .error .launch

theorem empty_chunks (classify : A → Bool) (input output : View) (before : M) :
    Lowering.allChunks ops classify input output 0 (List.range (chunkCount 0)) before=before := by
  norm_num [Lowering.allChunks,chunkCount,chunkSize,Lowering.chain]

theorem success (classify : A → Bool) (x y : Operand) (before : M) (savedDevice : Nat)
    (checked : checkInputs (.view x.metadata) (.view y.metadata)=.ok ())
    (launchesSucceed : Bool) (launched : x.metadata.n=0 ∨ launchesSucceed=Bool.true) :
    run ops classify x y before savedDevice launchesSucceed=.ok
      ⟨y,Lowering.allChunks ops classify x.metadata.view y.metadata.outputView x.metadata.n
        (List.range (chunkCount x.metadata.n)) before,savedDevice⟩ := by
  by_cases empty : x.metadata.n=0
  · simp [run,checked,empty,empty_chunks]
  · have ready := launched.resolve_left empty
    simp [run,checked,empty,ready]

def Calls [Inhabited A] (classify : A → Bool) (x y : ViewMetadata) : Prop :=
  ∀ c pid,pid<Finite.grid (chunkLength x.n c) → ∀ before,
    ∃ final,FP.Structural.exec (Source.algebra x.precision classify)
        (source x.precision (x.view.shift (chunkStart c)) (y.outputView.shift (chunkStart c))
          (chunkLength x.n c) 1024) (Source.backed ops before pid)=some final ∧
      (∀ q,Source.cellByte (final.mem "y" q)=ops.byte
        (Lowering.program ops classify (x.view.shift (chunkStart c)) (y.outputView.shift (chunkStart c))
          (chunkLength x.n c) pid before) q) ∧
      (∀ r q,r≠("y" : RegionName) → final.mem r q=(Source.backed ops before pid).mem r q)

theorem calls [Inhabited A] (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y) :
    Calls ops classify x y := by
  intro c pid launched before
  exact Source.program_refines ops classify x y h (chunkStart c) (chunkLength x.n c) pid
    (Finite.small x.n c) launched (Finite.active x.n c) before

theorem memory_complete (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y)
    (wordWidth : width=x.width) (before : M) :
    let after := Lowering.allChunks ops classify x.view y.outputView x.n (List.range (chunkCount x.n)) before
    (∀ i,i<x.n → ops.byte after (y.outputView.address i)=Memory.boolByte (classify (ops.read before (x.view.address i)))) ∧
    (∀ q,(∀ i,i<x.n → q≠y.outputView.address i) → ops.byte after q=ops.byte before q) ∧
    (∀ i,i<x.n → ops.read after (x.view.address i)=ops.read before (x.view.address i)) := by
  dsimp only
  rw [Lowering.complete ops classify x y h]
  exact Memory.complete ops classify x.view.address y.outputView.address x.n (Finite.batches x.n) before
    (Finite.coverage x.n) (fun i hi j hj => output_injective x y h i j hi hj)
    (by rw [wordWidth];exact Finite.physical_disjoint x y h)

theorem input_byte_frame (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y)
    (wordWidth : width=x.width) (before : M) (i k : Nat) (hi : i<x.n) (hk : k<x.width) :
    ops.byte (Lowering.allChunks ops classify x.view y.outputView x.n (List.range (chunkCount x.n)) before)
      (x.view.address i+k)=ops.byte before (x.view.address i+k) :=
  (memory_complete ops classify x y h wordWidth before).2.1 _
    (fun j hj => Finite.physical_disjoint x y h i hi k hk j hj)

/-- Return identity, declared view shape/dtype/device, exact output cells,
all exterior bytes, original inputs, and whole-source call refinements. -/
theorem complete [Inhabited A] (classify : A → Bool) (x y : Operand)
    (checked : checkInputs (.view x.metadata) (.view y.metadata)=.ok ())
    (wordWidth : width=x.metadata.width) (before : M) (savedDevice : Nat)
    (launchesSucceed : Bool) (launched : x.metadata.n=0 ∨ launchesSucceed=Bool.true) :
    InputContract x.metadata y.metadata ∧
    ∃ result,run ops classify x y before savedDevice launchesSucceed=.ok result ∧
      result.output=y ∧ result.currentDevice=savedDevice ∧
      (∀ i,i<x.metadata.n → ops.byte result.memory (y.metadata.outputView.address i)=
        Memory.boolByte (classify (ops.read before (x.metadata.view.address i)))) ∧
      (∀ q,(∀ i,i<x.metadata.n → q≠y.metadata.outputView.address i) → ops.byte result.memory q=ops.byte before q) ∧
      (∀ i,i<x.metadata.n → ops.read result.memory (x.metadata.view.address i)=ops.read before (x.metadata.view.address i)) ∧
      Calls ops classify x.metadata y.metadata := by
  have h : InputContract x.metadata y.metadata := (checkInputs_iff _ _).mp checked
  have done := memory_complete ops classify x.metadata y.metadata h wordWidth before
  exact ⟨h,_,success ops classify x y before savedDevice checked launchesSucceed launched,rfl,rfl,
    done.1,done.2.1,done.2.2,calls ops classify x.metadata y.metadata h⟩

theorem outcomes (classify : A → Bool) (x y : Operand) (before : M) (savedDevice : Nat) (ready : Bool) :
    (∃ e,run ops classify x y before savedDevice ready=.error (.input e)) ∨
    (run ops classify x y before savedDevice ready=.error .launch ∧
      InputContract x.metadata y.metadata ∧ x.metadata.n≠0 ∧ ready=Bool.false) ∨
    (∃ result,run ops classify x y before savedDevice ready=.ok result ∧
      InputContract x.metadata y.metadata ∧ result.output=y ∧ result.currentDevice=savedDevice) := by
  cases hc : checkInputs (.view x.metadata) (.view y.metadata) with
  | error e => exact Or.inl ⟨e,by simp [run,hc]⟩
  | ok u =>
    have good : InputContract x.metadata y.metadata := (checkInputs_iff _ _).mp hc
    by_cases empty : x.metadata.n=0
    · exact Or.inr (Or.inr ⟨⟨y,before,savedDevice⟩,by simp [run,hc,empty],good,rfl,rfl⟩)
    · cases ready
      · exact Or.inr (Or.inl ⟨by simp [run,hc,empty],good,empty,rfl⟩)
      · exact Or.inr (Or.inr ⟨⟨y,Lowering.allChunks ops classify x.metadata.view y.metadata.outputView x.metadata.n
          (List.range (chunkCount x.metadata.n)) before,savedDevice⟩,by simp [run,hc,empty],good,rfl,rfl⟩)

#axiomsClean complete
#axiomsClean outcomes
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel.Checked
