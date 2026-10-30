import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedCoverage
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedGridResult
import VeriTile.Triton.Float.ObservedProgramIO

/-! Canonical channels, logical windows, runtime bindings and complete launch
schedule, shared by Real and native interpretations. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelInterface
open VeriTile.Triton FP.Structural
noncomputable section

/-- Only strides feed this adapter; all Tensor bookkeeping is discarded. -/
def descriptor (strides : Fin 4 → Nat) : Tensor4 := ⟨0,0,0,0,strides⟩
def buffers (s : Shape) (incoming finalGradient : Fin 4 → Nat) : InputBuffers :=
  ⟨descriptor (contiguous4 s.B s.H s.T s.DK),
   descriptor (contiguous4 s.B s.H s.T s.DK),
   descriptor (contiguous4 s.B s.H s.T s.DV),descriptor incoming,
   descriptor (fun _=>0),descriptor finalGradient⟩

def layout (s : Shape) (incoming finalGradient : Fin 4 → Nat) : Quoted.Layout :=
  Quoted.layout s (buffers s incoming finalGradient)

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelInterface

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 18000
variable {α : Type}

abbrev inputPorts (s : Shape) : List IOChannel :=
  [⟨.real,[s.BH,s.T,s.DK],"q"⟩,⟨.real,[s.BH,s.T,s.DK],"k"⟩,
   ⟨.real,[s.BH,s.T,s.DV],"v"⟩,⟨.real,[s.BH,s.T,s.DV],"do"⟩,
   ⟨.real,[s.BH,s.DK,s.DV],"initial_state"⟩,⟨.real,[s.BH,s.DK,s.DV],"d_final_state"⟩]
abbrev outputPorts (s : Shape) : List IOChannel :=
  [⟨.real,[s.NK*rowCount s s.DV],"o"⟩,⟨.real,[s.NV*rowCount s s.DK],"dq"⟩,
   ⟨.real,[s.NV*rowCount s s.DK],"dk"⟩,⟨.real,[s.NK*rowCount s s.DV],"dv"⟩,
   ⟨.real,[s.BH*s.DK*s.DV],"final_state"⟩]

def inputWindow (s : Shape) (incoming finalGradient : Fin 4 → Nat)
    (i : Fin (inputPorts s).length) : TileIndex (inputPorts s)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun j => rowAddress s s.DK j.1.val j.2.1.val j.2.2.1.val
  | ⟨1,_⟩ => fun j => rowAddress s s.DK j.1.val j.2.1.val j.2.2.1.val
  | ⟨2,_⟩ => fun j => rowAddress s s.DV j.1.val j.2.1.val j.2.2.1.val
  | ⟨3,_⟩ => fun j => (j.1.val/s.H)*incoming 0+(j.1.val%s.H)*incoming 1+
      j.2.2.1.val*incoming 3+j.2.1.val*incoming 2
  | ⟨4,_⟩ => fun j => stateOffset s j.1.val j.2.1.val j.2.2.1.val
  | ⟨5,_⟩ => fun j => (j.1.val/s.H)*finalGradient 0+(j.1.val%s.H)*finalGradient 1+
      j.2.1.val*finalGradient 2+j.2.2.1.val*finalGradient 3

def inputActive (mode : Mode) (seed final : Bool) (i : Fin 6) : Prop :=
  i.val<3 ∨ (i.val=3 ∧ mode=.backward) ∨ (i.val=4 ∧ seed=Bool.true) ∨
    (i.val=5 ∧ mode=.backward ∧ final=Bool.true)

def outputWindow (s : Shape) (i : Fin (outputPorts s).length) : TileIndex (outputPorts s)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => fun j => j.1.val

def outputActive (mode : Mode) (final : Bool) (i : Fin 5) : Prop :=
  (mode=.forward ∧ (i.val=0 ∨ (i.val=4 ∧ final=Bool.true))) ∨
    (mode=.backward ∧ (i.val=1 ∨ i.val=2 ∨ i.val=3))

def kind : Fin 5 → PartialKind
  | ⟨0,_⟩ => .output | ⟨1,_⟩ => .dq | ⟨2,_⟩ => .dk | ⟨3,_⟩ => .dv | ⟨4,_⟩ => .finalState

def job (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) (pid : ProgramId) : Invocation (Staged.Node α) where
  kernel := Quoted.kernel mode s (KernelInterface.buffers s incoming finalGradient) seed final
  backend := Staged.algebra a p.plans
  arguments := [⟨"scale",.real,[],fun _ => Staged.tagged .scale (a.scale s.DK)⟩]
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then pid.iv else if ax=1 then pid.ik else pid.head

/-- Input nodes retain their full original carrier. Only the final observation
forgets expression tags; no leaf-only input restriction is introduced. -/
abbrev io (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    ProgramIO (Staged.Node α) (Staged.Node α) where
  inputs := inputPorts s
  outputs := outputPorts s
  scratch := []
  buffers := ["q","k","v","do","initial_state","d_final_state","o","dq","dk","dv","final_state"]
  read := fun i _ => inputWindow s incoming finalGradient i
  inputMask := fun i _ _ => inputActive mode seed final i
  write := fun i _ => outputWindow s i
  outputMask := fun i _ _ => outputActive mode final i
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => (gridPrograms s).map (job mode a s p incoming finalGradient seed final)
  encode := id
  observe := fun _ _ => id

/-- A logical 3-D input observation with a harmless value outside its shape.
All target recurrences mask those coordinates before using them. -/
def array (zero : α) (shape0 shape1 shape2 : Nat)
    (xs : TileIndex [shape0,shape1,shape2] → Staged.Node α) (h t c : Nat) : α :=
  if hh : h<shape0 then if ht : t<shape1 then if hc : c<shape2 then
    (xs (⟨h,hh⟩,⟨t,ht⟩,⟨c,hc⟩,PUnit.unit)).value else zero else zero else zero

def logicalInputs (a : Arithmetic α) (s : Shape) (xs : ChannelValues (Staged.Node α) (inputPorts s)) : Inputs α where
  q := array a.zero s.BH s.T s.DK (xs 0)
  k := array a.zero s.BH s.T s.DK (xs 1)
  v := array a.zero s.BH s.T s.DV (xs 2)
  do_ := array a.zero s.BH s.T s.DV (xs 3)
  seed := array a.zero s.BH s.DK s.DV (xs 4)
  finalGrad := array a.zero s.BH s.DK s.DV (xs 5)

def expected (a : Arithmetic α) (s : Shape) (p : PlanBinding s) (seed final : Bool)
    (xs : ChannelValues (Staged.Node α) (inputPorts s)) : ChannelValues α (outputPorts s) :=
  fun i => match i with
  | ⟨0,_⟩ => fun j => partialValue a s p.plans (logicalInputs a s xs) seed final .output j.1.val
  | ⟨1,_⟩ => fun j => partialValue a s p.plans (logicalInputs a s xs) seed final .dq j.1.val
  | ⟨2,_⟩ => fun j => partialValue a s p.plans (logicalInputs a s xs) seed final .dk j.1.val
  | ⟨3,_⟩ => fun j => partialValue a s p.plans (logicalInputs a s xs) seed final .dv j.1.val
  | ⟨4,_⟩ => fun j => partialValue a s p.plans (logicalInputs a s xs) seed final .finalState j.1.val

#specNonCircular expected avoiding [Quoted.gridExec,FP.Structural.exec,sourceCell,programO,programQ,programK,programV]
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section

structure TensorInputs (α : Type) (c : Shape) where
  q : Fin (c.BH) → Fin (c.T) → Fin (c.DK) → α
  k : Fin (c.BH) → Fin (c.T) → Fin (c.DK) → α
  v : Fin (c.BH) → Fin (c.T) → Fin (c.DV) → α
  incoming : Fin (c.BH) → Fin (c.T) → Fin (c.DV) → α
  initialState : Fin (c.BH) → Fin (c.DK) → Fin (c.DV) → α
  finalGradient : Fin (c.BH) → Fin (c.DK) → Fin (c.DV) → α

def inputCoordinates (α : Type) (c : Shape) : ChannelValues α (inputPorts c) ≃ TensorInputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.BH, c.T, c.DK] (xs (0 : Fin 6)),
    Values.curryEquiv α .real [c.BH, c.T, c.DK] (xs (1 : Fin 6)),
    Values.curryEquiv α .real [c.BH, c.T, c.DV] (xs (2 : Fin 6)),
    Values.curryEquiv α .real [c.BH, c.T, c.DV] (xs (3 : Fin 6)),
    Values.curryEquiv α .real [c.BH, c.DK, c.DV] (xs (4 : Fin 6)),
    Values.curryEquiv α .real [c.BH, c.DK, c.DV] (xs (5 : Fin 6))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.BH, c.T, c.DK]).symm xs.q) (Fin.cases ((Values.curryEquiv α .real [c.BH, c.T, c.DK]).symm xs.k) (Fin.cases ((Values.curryEquiv α .real [c.BH, c.T, c.DV]).symm xs.v) (Fin.cases ((Values.curryEquiv α .real [c.BH, c.T, c.DV]).symm xs.incoming) (Fin.cases ((Values.curryEquiv α .real [c.BH, c.DK, c.DV]).symm xs.initialState) (Fin.cases ((Values.curryEquiv α .real [c.BH, c.DK, c.DV]).symm xs.finalGradient) (fun i => Fin.elim0 i)))))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.BH, c.T, c.DK]).symm_apply_apply (xs (0 : Fin 6))
    · exact (Values.curryEquiv α .real [c.BH, c.T, c.DK]).symm_apply_apply (xs (1 : Fin 6))
    · exact (Values.curryEquiv α .real [c.BH, c.T, c.DV]).symm_apply_apply (xs (2 : Fin 6))
    · exact (Values.curryEquiv α .real [c.BH, c.T, c.DV]).symm_apply_apply (xs (3 : Fin 6))
    · exact (Values.curryEquiv α .real [c.BH, c.DK, c.DV]).symm_apply_apply (xs (4 : Fin 6))
    · exact (Values.curryEquiv α .real [c.BH, c.DK, c.DV]).symm_apply_apply (xs (5 : Fin 6))
  right_inv := by intro xs; cases xs; rfl

/-- Every flat partial position corresponds to one split/head/time/channel. -/
def partialIndex (s : Shape) (count width : Nat) :
    Fin (count * rowCount s width) ≃ (Fin count × Fin s.BH × Fin s.T × Fin width) where
  toFun := fun i =>
    let h := unflatten_partial s count width i.val i.isLt
    (⟨flatSplit s width i.val,h.1⟩,⟨flatHead s width i.val,h.2.1⟩,
      ⟨flatTime s width i.val,h.2.2.1⟩,⟨flatChannel width i.val,h.2.2.2.1⟩)
  invFun := fun j => ⟨partialOffset s width j.1.val j.2.1.val j.2.2.1.val j.2.2.2.val,
    partialOffset_bound s count width _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.isLt⟩
  left_inv := by
    intro i
    apply Fin.ext
    exact (unflatten_partial s count width i.val i.isLt).2.2.2.2
  right_inv := by
    rintro ⟨part,head,time,channel⟩
    have h := flat_partial s width part.val head.val time.val channel.val head.isLt time.isLt channel.isLt
    exact Prod.ext (Fin.ext h.1) (Prod.ext (Fin.ext h.2.1)
      (Prod.ext (Fin.ext h.2.2.1) (Fin.ext h.2.2.2)))

def stateIndex (s : Shape) :
    Fin (s.BH * s.DK * s.DV) ≃ (Fin s.BH × Fin s.DK × Fin s.DV) where
  toFun := fun i =>
    let h := unflatten_state s i.val i.isLt
    (⟨i.val/s.DV/s.DK,h.1⟩,⟨i.val/s.DV%s.DK,h.2.1⟩,⟨i.val%s.DV,h.2.2.1⟩)
  invFun := fun j => ⟨stateOffset s j.1.val j.2.1.val j.2.2.val,
    stateOffset_bound s _ _ _ j.1.isLt j.2.1.isLt j.2.2.isLt⟩
  left_inv := by intro i; exact Fin.ext (unflatten_state s i.val i.isLt).2.2.2
  right_inv := by
    rintro ⟨head,key,value⟩
    apply Prod.ext
    · apply Fin.ext
      simp only [stateOffset, pair_div s.DV _ value.val value.isLt,
        pair_div s.DK _ key.val key.isLt]
    · apply Prod.ext
      · apply Fin.ext
        simp only [stateOffset, pair_div s.DV _ value.val value.isLt,
          pair_mod s.DK _ key.val key.isLt]
      · apply Fin.ext
        simp only [stateOffset, pair_mod s.DV _ value.val value.isLt]

def partialValues (α : Type) (s : Shape) (count width : Nat) :
    Values α .real [count * rowCount s width] ≃
      (Fin count → Fin s.BH → Fin s.T → Fin width → α) where
  toFun := fun xs part head time channel => xs ((partialIndex s count width).symm (part,head,time,channel),())
  invFun := fun xs j =>
    let i := partialIndex s count width j.1
    xs i.1 i.2.1 i.2.2.1 i.2.2.2
  left_inv := by
    intro xs
    funext j
    change xs ((partialIndex s count width).symm (partialIndex s count width j.1),()) = xs j
    rw [Equiv.symm_apply_apply]
  right_inv := by
    intro xs
    funext part head time channel
    change (let i := partialIndex s count width ((partialIndex s count width).symm (part,head,time,channel));
      xs i.1 i.2.1 i.2.2.1 i.2.2.2) = xs part head time channel
    rw [Equiv.apply_symm_apply]

def stateValues (α : Type) (s : Shape) : Values α .real [s.BH*s.DK*s.DV] ≃
    (Fin s.BH → Fin s.DK → Fin s.DV → α) where
  toFun := fun xs head key value => xs ((stateIndex s).symm (head,key,value),())
  invFun := fun xs j => let i := stateIndex s j.1; xs i.1 i.2.1 i.2.2
  left_inv := by
    intro xs
    funext j
    change xs ((stateIndex s).symm (stateIndex s j.1),()) = xs j
    rw [Equiv.symm_apply_apply]
  right_inv := by
    intro xs
    funext head key value
    change (let i := stateIndex s ((stateIndex s).symm (head,key,value)); xs i.1 i.2.1 i.2.2) = xs head key value
    rw [Equiv.apply_symm_apply]

structure TensorOutputs (α : Type) (s : Shape) where
  output : Fin s.NK → Fin s.BH → Fin s.T → Fin s.DV → α
  dq : Fin s.NV → Fin s.BH → Fin s.T → Fin s.DK → α
  dk : Fin s.NV → Fin s.BH → Fin s.T → Fin s.DK → α
  dv : Fin s.NK → Fin s.BH → Fin s.T → Fin s.DV → α
  finalState : Fin s.BH → Fin s.DK → Fin s.DV → α

def outputCoordinates (α : Type) (s : Shape) : ChannelValues α (outputPorts s) ≃ TensorOutputs α s where
  toFun := fun xs => ⟨partialValues α s s.NK s.DV (xs 0),partialValues α s s.NV s.DK (xs 1),
    partialValues α s s.NV s.DK (xs 2),partialValues α s s.NK s.DV (xs 3),stateValues α s (xs 4)⟩
  invFun := fun xs => Fin.cases ((partialValues α s s.NK s.DV).symm xs.output)
    (Fin.cases ((partialValues α s s.NV s.DK).symm xs.dq)
      (Fin.cases ((partialValues α s s.NV s.DK).symm xs.dk)
        (Fin.cases ((partialValues α s s.NK s.DV).symm xs.dv)
          (Fin.cases ((stateValues α s).symm xs.finalState) (fun i => Fin.elim0 i)))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (partialValues α s s.NK s.DV).symm_apply_apply (xs 0)
    · exact (partialValues α s s.NV s.DK).symm_apply_apply (xs 1)
    · exact (partialValues α s s.NV s.DK).symm_apply_apply (xs 2)
    · exact (partialValues α s s.NK s.DV).symm_apply_apply (xs 3)
    · exact (stateValues α s).symm_apply_apply (xs 4)
  right_inv := by
    intro xs
    change TensorOutputs.mk
      ((partialValues α s s.NK s.DV) ((partialValues α s s.NK s.DV).symm xs.output))
      ((partialValues α s s.NV s.DK) ((partialValues α s s.NV s.DK).symm xs.dq))
      ((partialValues α s s.NV s.DK) ((partialValues α s s.NV s.DK).symm xs.dk))
      ((partialValues α s s.NK s.DV) ((partialValues α s s.NK s.DV).symm xs.dv))
      ((stateValues α s) ((stateValues α s).symm xs.finalState)) = xs
    simp only [Equiv.apply_symm_apply]

def tensorData (s : Shape) (xs : TensorInputs (Staged.Node ℝ) s) : Inputs ℝ :=
  logicalInputs realArithmetic s ((inputCoordinates (Staged.Node ℝ) s).symm xs)

/-- Exact original observation, with only its finite coordinates transported.
Arbitrary input tags and all five optional/masked outputs remain in the judgment. -/
def logicalIO (mode : Mode) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    LogicalIO (ChannelValues (Staged.Node ℝ) (inputPorts s)) (ChannelValues ℝ (outputPorts s))
      (TensorInputs (Staged.Node ℝ) s) (TensorOutputs ℝ s)
      (fun f => (io mode realArithmetic s p incoming finalGradient seed final).Observes
        (fun _ _ n => n.value) f) where
  inputs := inputCoordinates (Staged.Node ℝ) s
  outputs := outputCoordinates ℝ s
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
