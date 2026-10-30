import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Real
import bench.repairs.tritonbench_g.geglu_tanh_triton.Kernels
import VeriTile.Triton.Float.ProgramIO

/-! Complete source grids, including backward's in-place A/B outputs. The
selected source fixes all loads, operations, constants and stores to the
original FP32 route. Structural uses `.real` for these typed cell channels. -/
namespace VeriTile.Bench.Repairs.GegluTanhTriton.IO
open VeriTile Triton FP.Structural
noncomputable section

abbrev inputs (count : Nat) : List IOChannel :=
  [⟨.real,[count],"A"⟩,⟨.real,[count],"B"⟩,⟨.real,[count],"DC"⟩]
abbrev outputs (count : Nat) : List IOChannel :=
  [⟨.real,[count],"C"⟩,⟨.real,[count],"A"⟩,⟨.real,[count],"B"⟩]

def channel (o : Fin 3) : Channel :=
  Fin.cases .c (Fin.cases .da (Fin.cases .db (fun i=>Fin.elim0 i))) o

def launch (M : Algebra β) (candidate backward : Bool) (count pid : Nat) : Invocation β where
  kernel:=selectedSource backward candidate 128 count
  backend:=M
  pids:=fun ax=>if ax=0 then pid else 0

def program (M : Algebra β) (candidate backward : Bool) (count : Nat) (order : List Nat)
    (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs:=inputs count
  outputs:=outputs count
  scratch:=[]
  buffers:=["A","B","C","DC"]
  read:=fun i=>Fin.cases (fun _ j=>j.1.val)
    (Fin.cases (fun _ j=>j.1.val) (Fin.cases (fun _ j=>j.1.val) (fun i=>Fin.elim0 i))) i
  inputMask:=fun i _ _=>i.val≠2 ∨ backward=Bool.true
  write:=fun o=>Fin.cases (fun _ j=>j.1.val)
    (Fin.cases (fun _ j=>j.1.val) (Fin.cases (fun _ j=>j.1.val) (fun i=>Fin.elim0 i))) o
  outputMask:=fun o _ _=>enabled backward (channel o)
  scratchWrite:=fun i=>Fin.elim0 i
  scratchMask:=fun i=>Fin.elim0 i
  schedule:=fun _=>order.map (launch M candidate backward count)
  encode:=encode
  observe:=fun _ _=>observe

end
end VeriTile.Bench.Repairs.GegluTanhTriton.IO

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Reference
open VeriTile Triton FP.Structural
noncomputable section

def realIO (candidate backward : Bool) (count : Nat) (order : List Nat) : ProgramIO ℝ ℝ :=
  IO.program RealExecution.algebra candidate backward count order id id

abbrev inputs (count : Nat) : List IOChannel := IO.inputs count
abbrev outputs (count : Nat) : List IOChannel := IO.outputs count

def formula [Inhabited α] (backward : Bool) (count : Nat) (f : Channel → α → α → α → α) :
    ChannelValues α (inputs count) → ChannelValues α (outputs count) :=
  fun vals o=>Fin.cases
    (fun j=>f .c (vals ⟨0,by simp [inputs, IO.inputs]⟩ j) (vals ⟨1,by simp [inputs, IO.inputs]⟩ j) (if backward then vals ⟨2,by simp [inputs, IO.inputs]⟩ j else default))
    (Fin.cases (fun j=>f .da (vals ⟨0,by simp [inputs, IO.inputs]⟩ j) (vals ⟨1,by simp [inputs, IO.inputs]⟩ j) (if backward then vals ⟨2,by simp [inputs, IO.inputs]⟩ j else default))
      (Fin.cases (fun j=>f .db (vals ⟨0,by simp [inputs, IO.inputs]⟩ j) (vals ⟨1,by simp [inputs, IO.inputs]⟩ j) (if backward then vals ⟨2,by simp [inputs, IO.inputs]⟩ j else default)) (fun i=>Fin.elim0 i))) o

end
end VeriTile.Bench.Repairs.GegluTanhTriton.Reference

namespace VeriTile.Bench.Repairs.GegluTanhTriton.IO
open VeriTile Triton FP.Structural
noncomputable section
structure TensorInputs (α : Type) (count : Nat) where
  a : Fin (count) → α
  b : Fin (count) → α
  incoming : Fin (count) → α

def inputCoordinates (α : Type) (count : Nat) : ChannelValues α (Reference.inputs count) ≃ TensorInputs α count where
  toFun := fun xs => ⟨Values.curryEquiv α .real [count] (xs (0 : Fin 3)),
    Values.curryEquiv α .real [count] (xs (1 : Fin 3)),
    Values.curryEquiv α .real [count] (xs (2 : Fin 3))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [count]).symm xs.a) (Fin.cases ((Values.curryEquiv α .real [count]).symm xs.b) (Fin.cases ((Values.curryEquiv α .real [count]).symm xs.incoming) (fun i => Fin.elim0 i))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [count]).symm_apply_apply (xs (0 : Fin 3))
    · exact (Values.curryEquiv α .real [count]).symm_apply_apply (xs (1 : Fin 3))
    · exact (Values.curryEquiv α .real [count]).symm_apply_apply (xs (2 : Fin 3))
  right_inv := by intro xs; cases xs; rfl

def outputCoordinates (α : Type) (count : Nat) : ChannelValues α (Reference.outputs count) ≃ (Channel → Fin count → α) where
  toFun := fun xs ch i => match ch with
    | .c => xs (0 : Fin 3) (i,()) | .da => xs (1 : Fin 3) (i,()) | .db => xs (2 : Fin 3) (i,())
  invFun := fun xs => Fin.cases (fun j => xs .c j.1)
    (Fin.cases (fun j => xs .da j.1) (Fin.cases (fun j => xs .db j.1) (fun i => Fin.elim0 i)))
  left_inv := by
    intro xs
    funext i j
    fin_cases i <;> cases j with | mk a b => cases b; rfl
  right_inv := by intro xs; funext ch; cases ch <;> rfl

def logicalIO (candidate backward : Bool) (count : Nat) (order : List Nat) :
    LogicalProgramIO ℝ ℝ (TensorInputs ℝ count) (Channel → Fin count → ℝ) where
  core := Reference.realIO candidate backward count order
  inputs := inputCoordinates ℝ count
  outputs := outputCoordinates ℝ count
end
end VeriTile.Bench.Repairs.GegluTanhTriton.IO
