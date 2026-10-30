import bench.tritonbench_g.bgmv_shrink_kernel.Kernels
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.FPWorkset
import VeriTile.Triton.Float.PerAddressProgramIO
import VeriTile.Triton.Memory.LogicalIO
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.Real

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IO
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section

/-- A finite channel covers every selected slot without assuming that an
arbitrary kernel-level metadata value is below the wrapper's adapter count. -/
def slots (h : Call) (indices : Nat → Int) : Nat :=
  (Finset.univ.sup fun b : Fin h.batches => (indices b.val).toNat)+1

theorem selected_inside (h : Call) (indices : Nat → Int) (b : Fin h.batches) :
    (indices b.val).toNat < slots h indices :=
  Nat.lt_succ_of_le (Finset.le_sup (f := fun b : Fin h.batches => (indices b.val).toNat) (Finset.mem_univ b))

@[reducible] def inputs (h : Call) (indices : Nat → Int) : List IOChannel :=
  [⟨.int,[h.batches],"Indices"⟩,
   ⟨(dtype h.storage).toTileDType,[h.batches,h.K],"X"⟩,
   ⟨(dtype h.storage).toTileDType,[slots h indices,h.N,h.K],"W"⟩,
   ⟨h.outputType.toTileDType,[h.batches,h.N],"Y"⟩]

@[reducible] def outputs (h : Call) : List IOChannel :=
  [⟨h.outputType.toTileDType,[h.batches,h.N],"Y"⟩]

/-- The full launch catalog includes every batch/split CTA even when N=0. -/
def programs (h : Call) : List (Program h) :=
  (List.finRange h.batches).flatMap fun b => (List.finRange 64).map fun split => ⟨b,split⟩

@[simp] theorem program_mem (h : Call) (p : Program h) : p ∈ programs h := by
  rcases p with ⟨b,split⟩
  simp [programs]

def invocation (M : Algebra α) (P : Primitives α) (h : Call) (scale : α)
    (p : Program h) : Invocation α where
  kernel := Surface.source h
  backend := machine M P
  arguments := [⟨"scaling",.real,[],fun _ => scale⟩]
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then p.split.val else if ax=1 then p.batch.val else 0
  numPids := some (fun ax => if ax=0 then 64 else if ax=1 then h.batches else 1)

/-- Source identity, full launch catalog, actual per-address event order,
selected adapter windows, in-place Y and runtime scaling are explicit data.
The X mask is independent of N because an active CTA still loads X at N=0. -/
@[reducible] def programIO (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h)
    (observe : α → γ) : PerAddressProgramIO α α γ (Program h) where
  inputs := inputs h indices
  outputs := outputs h
  scratch := []
  buffers := ["Indices","X","W","Y"]
  read := Fin.cases (fun _ j => j.1.val)
    (Fin.cases (fun _ j => inputOffset h j.1.val j.2.1.val)
      (Fin.cases (fun _ j => weightOffset h j.1.val j.2.1.val j.2.2.1.val)
        (Fin.cases (fun _ j => outputOffset h j.1.val j.2.1.val) (fun i => Fin.elim0 i))))
  inputMask := Fin.cases (fun _ _ => True)
    (Fin.cases (fun _ j => indices j.1.val ≠ -1)
      (Fin.cases (fun _ j => ∃ b : Fin h.batches,
        indices b.val ≠ -1 ∧ (indices b.val).toNat = j.1.val)
        (Fin.cases (fun _ _ => True) (fun i => Fin.elim0 i))))
  write := Fin.cases (fun _ j => outputOffset h j.1.val j.2.1.val) (fun i => Fin.elim0 i)
  outputMask := Fin.cases (fun _ _ => True) (fun i => Fin.elim0 i)
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  programs := fun _ => programs h
  invocation := fun _ => invocation M P h scale
  events := fun _ => Fin.cases (fun j =>
    if indices j.1.val = -1 then [] else
      (List.finRange 64).map fun split => ⟨j.1,split⟩) (fun i => Fin.elim0 i)
  order := fun _ => Fin.cases (fun j =>
    if indices j.1.val = -1 then [] else
      (orders.order j.1 j.2.1).map fun split => ⟨j.1,split⟩) (fun i => Fin.elim0 i)
  encode := id
  observe := fun _ _ => observe

/-- Metadata is checked independently of the arithmetic carrier and source.
No adapter-count restriction is added at kernel level. -/
def domain (h : Call) (indices : Nat → Int) (vals : ChannelValues α (inputs h indices)) : Prop :=
  (∀ b : Fin h.batches, vals 0 (b,PUnit.unit) = indices b.val) ∧
  (∀ b : Fin h.batches, indices b.val = -1 ∨ 0 ≤ indices b.val)

#axiomsClean selected_inside
#axiomsClean program_mem
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IO

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IO
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section

structure LogicalInputs (h : Call) (indices : Nat → Int) where
  metadata : Fin h.batches → Int
  input : Fin h.batches → Fin h.K → Value ℝ (dtype h.storage).toTileDType
  weight : Fin (slots h indices) → Fin h.N → Fin h.K → Value ℝ (dtype h.storage).toTileDType
  previous : Fin h.batches → Fin h.N → Value ℝ h.outputType.toTileDType

def logicalInputsEquiv (h : Call) (indices : Nat → Int) :
    ChannelValues ℝ (inputs h indices) ≃ LogicalInputs h indices where
  toFun := fun vals => ⟨fun b => vals 0 (b,()), fun b k => vals 1 (b,k,()),
    fun slot row k => vals 2 (slot,row,k,()), fun b row => vals 3 (b,row,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.metadata j.1)
    (Fin.cases (fun j => xs.input j.1 j.2.1)
      (Fin.cases (fun j => xs.weight j.1 j.2.1 j.2.2.1)
        (Fin.cases (fun j => xs.previous j.1 j.2.1) (fun i => i.elim0))))
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    · rcases j with ⟨b,⟨⟩⟩; rfl
    · rcases j with ⟨b,k,⟨⟩⟩; rfl
    · rcases j with ⟨s,r,k,⟨⟩⟩; rfl
    · rcases j with ⟨b,r,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputsEquiv (h : Call) : ChannelValues ℝ (outputs h) ≃
    (Fin h.batches → Fin h.N → Value ℝ h.outputType.toTileDType) :=
  (ChannelValues.singleEquiv ℝ _).trans
    (Values.matrixEquiv ℝ h.outputType.toTileDType h.batches h.N)

def LogicalInputs.selectedWeight {h : Call} {indices : Nat → Int}
    (xs : LogicalInputs h indices) (b : Fin h.batches) (row : Fin h.N) (k : Fin h.K) :=
  xs.weight ⟨(indices b.val).toNat,selected_inside h indices b⟩ row k

def logicalDomain (h : Call) (indices : Nat → Int) (xs : LogicalInputs h indices) : Prop :=
  (∀ b, xs.metadata b=indices b.val) ∧ (∀ b : Fin h.batches, indices b.val = -1 ∨ 0 ≤ indices b.val)

def logicalIO (h : Call) (indices : Nat → Int) (orders : Orders h) :
    LogicalDomainIO (ChannelValues ℝ (inputs h indices)) (ChannelValues ℝ (outputs h))
      (LogicalInputs h indices) (Fin h.batches → Fin h.N → Value ℝ h.outputType.toTileDType)
      (programIO Real.algebra Real.primitives h indices h.scaling orders id).ImplementsOnCells where
  inputs := logicalInputsEquiv h indices
  outputs := logicalOutputsEquiv h

end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IO
