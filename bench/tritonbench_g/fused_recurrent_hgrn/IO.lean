import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Reference
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.IO
open VeriTile.Triton FP.Structural
noncomputable section

/-- FP32 source channels use the Structural interpreter's erased Real cells.
The quoted sources retain every FP32 load, arithmetic operation and store. -/
abbrev inputs (BH T D : Nat) : List IOChannel :=
  [⟨.real,[BH,T,D],"x"⟩,⟨.real,[BH,T,D],"g"⟩,⟨.real,[BH,T,D],"do"⟩,
   ⟨.real,[BH,D],"h0"⟩,⟨.real,[BH,D],"dht"⟩]
abbrev outputs (BH T D : Nat) : List IOChannel :=
  [⟨.real,[BH,T,D],"o"⟩,⟨.real,[BH,T,D],"dx"⟩,⟨.real,[BH,T,D],"dg"⟩,
   ⟨.real,[BH,D],"ht"⟩]

def invocation (p : Ops α) (source : ComputeKernel) (id : Nat × Nat) : Invocation α where
  kernel := source
  backend := algebra p
  pids := fun ax => if ax=0 then id.1 else id.2
  setPids := fun ax => decide (ax=0 ∨ ax=1)

/-- The wrapper's complete two-dimensional grid, with independent forward
and backward block sizes and the actual phase order. -/
def programs (BH D BD : Nat) : List (Nat × Nat) :=
  (List.range ((D+BD-1)/BD)).flatMap (fun block => (List.range BH).map (fun bh => (block,bh)))

abbrev recurrentIO (p : Ops α) (observe : α → α) (BH T D BDF BDB : Nat)
    (U S V : Bool) : ProgramIO α α where
  inputs := inputs BH T D
  outputs := outputs BH T D
  scratch := []
  buffers := ["x","g","do","h0","dht","o","dx","dg","ht"]
  read := fun i _ => match i with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun j => rowAddr j.1.val T D j.2.1.val j.2.2.1.val
    | ⟨3,_⟩ | ⟨4,_⟩ => fun j => stateAddr j.1.val D j.2.1.val
    | ⟨_+5,h⟩ => False.elim (by simp at h)
  inputMask := fun i _ _ => match i.val with
    | 3 => U=Bool.true
    | 4 => V=Bool.true
    | _ => True
  write := fun o _ => match o with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun j => rowAddr j.1.val T D j.2.1.val j.2.2.1.val
    | ⟨3,_⟩ => fun j => stateAddr j.1.val D j.2.1.val
    | ⟨_+4,h⟩ => False.elim (by simp at h)
  outputMask := fun o _ _ => o.val≠3 ∨ S=Bool.true
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ =>
    (programs BH D BDF).map (invocation p (forward T D BDF U S)) ++
    (programs BH D BDB).map (invocation p (backward T D BDB U V))
  encode := id
  observe := fun _ _ => observe

/-- Extend finite logical channels only to write independent recurrences.
The source never reads the extension outside the declared active windows. -/
def asInput (p : Ops α) (BH T D : Nat) (vals : ChannelValues α (inputs BH T D)) : Input α :=
  fun r bh t d =>
    if hb : bh<BH then if hd : d<D then
      if r="h0" then vals 3 (⟨bh,hb⟩,⟨d,hd⟩,())
      else if r="dht" then vals 4 (⟨bh,hb⟩,⟨d,hd⟩,())
      else if ht : t<T then
        if r="x" then vals 0 (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,())
        else if r="g" then vals 1 (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,())
        else if r="do" then vals 2 (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,())
        else p.zero
      else p.zero
    else p.zero else p.zero

/-- Logical output tuple: all saved states, both input gradients and the
optional final state. The mathematical/native functions are supplied by readers. -/
def result (BH T D : Nat) (forwardState : Nat → Nat → Nat → α)
    (gradX gradG : Nat → Nat → Nat → α) : ChannelValues α (outputs BH T D) :=
  fun o => match o with
    | ⟨0,_⟩ => fun j => forwardState j.1.val j.2.2.1.val (j.2.1.val+1)
    | ⟨1,_⟩ => fun j => gradX j.1.val j.2.2.1.val j.2.1.val
    | ⟨2,_⟩ => fun j => gradG j.1.val j.2.2.1.val j.2.1.val
    | ⟨3,_⟩ => fun j => forwardState j.1.val j.2.1.val T
    | ⟨_+4,h⟩ => False.elim (by simp at h)

end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.IO

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.IO
open VeriTile.Triton FP.Structural
noncomputable section
structure TensorInputs (α : Type) (BH T D : Nat) where
  x : Fin (BH) → Fin (T) → Fin (D) → α
  gate : Fin (BH) → Fin (T) → Fin (D) → α
  incoming : Fin (BH) → Fin (T) → Fin (D) → α
  initialState : Fin (BH) → Fin (D) → α
  terminalGradient : Fin (BH) → Fin (D) → α

def inputCoordinates (α : Type) (BH T D : Nat) : ChannelValues α (inputs BH T D) ≃ TensorInputs α BH T D where
  toFun := fun xs => ⟨Values.curryEquiv α .real [BH, T, D] (xs (0 : Fin 5)),
    Values.curryEquiv α .real [BH, T, D] (xs (1 : Fin 5)),
    Values.curryEquiv α .real [BH, T, D] (xs (2 : Fin 5)),
    Values.curryEquiv α .real [BH, D] (xs (3 : Fin 5)),
    Values.curryEquiv α .real [BH, D] (xs (4 : Fin 5))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [BH, T, D]).symm xs.x) (Fin.cases ((Values.curryEquiv α .real [BH, T, D]).symm xs.gate) (Fin.cases ((Values.curryEquiv α .real [BH, T, D]).symm xs.incoming) (Fin.cases ((Values.curryEquiv α .real [BH, D]).symm xs.initialState) (Fin.cases ((Values.curryEquiv α .real [BH, D]).symm xs.terminalGradient) (fun i => Fin.elim0 i))))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [BH, T, D]).symm_apply_apply (xs (0 : Fin 5))
    · exact (Values.curryEquiv α .real [BH, T, D]).symm_apply_apply (xs (1 : Fin 5))
    · exact (Values.curryEquiv α .real [BH, T, D]).symm_apply_apply (xs (2 : Fin 5))
    · exact (Values.curryEquiv α .real [BH, D]).symm_apply_apply (xs (3 : Fin 5))
    · exact (Values.curryEquiv α .real [BH, D]).symm_apply_apply (xs (4 : Fin 5))
  right_inv := by intro xs; cases xs; rfl

structure TensorOutputs (α : Type) (BH T D : Nat) where
  savedState : Fin (BH) → Fin (T) → Fin (D) → α
  dx : Fin (BH) → Fin (T) → Fin (D) → α
  dg : Fin (BH) → Fin (T) → Fin (D) → α
  finalState : Fin (BH) → Fin (D) → α

def outputCoordinates (α : Type) (BH T D : Nat) : ChannelValues α (outputs BH T D) ≃ TensorOutputs α BH T D where
  toFun := fun xs => ⟨Values.curryEquiv α .real [BH, T, D] (xs (0 : Fin 4)),
    Values.curryEquiv α .real [BH, T, D] (xs (1 : Fin 4)),
    Values.curryEquiv α .real [BH, T, D] (xs (2 : Fin 4)),
    Values.curryEquiv α .real [BH, D] (xs (3 : Fin 4))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [BH, T, D]).symm xs.savedState) (Fin.cases ((Values.curryEquiv α .real [BH, T, D]).symm xs.dx) (Fin.cases ((Values.curryEquiv α .real [BH, T, D]).symm xs.dg) (Fin.cases ((Values.curryEquiv α .real [BH, D]).symm xs.finalState) (fun i => Fin.elim0 i)))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [BH, T, D]).symm_apply_apply (xs (0 : Fin 4))
    · exact (Values.curryEquiv α .real [BH, T, D]).symm_apply_apply (xs (1 : Fin 4))
    · exact (Values.curryEquiv α .real [BH, T, D]).symm_apply_apply (xs (2 : Fin 4))
    · exact (Values.curryEquiv α .real [BH, D]).symm_apply_apply (xs (3 : Fin 4))
  right_inv := by intro xs; cases xs; rfl


def tensorData (BH T D : Nat) (xs : TensorInputs ℝ BH T D) : Input ℝ :=
  asInput realOps BH T D ((inputCoordinates ℝ BH T D).symm xs)

def logicalIO (BH T D BDF BDB : Nat) (U S V : Bool) :
    LogicalProgramIO ℝ ℝ (TensorInputs ℝ BH T D) (TensorOutputs ℝ BH T D) where
  core := recurrentIO realOps id BH T D BDF BDB U S V
  inputs := inputCoordinates ℝ BH T D
  outputs := outputCoordinates ℝ BH T D
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.IO
