import VeriTile.Triton.Float.ProgramEquivalence
import bench.tritonbench_g.bgmv_expand_slice.Proofs.KernelRaw
import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.bgmv_expand_slice.Real

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
variable {A : Type} [Inhabited A]

def invocation (M : Algebra A) (R : Reduction A) (h : Call) (batch row : Nat) : Invocation A where
  kernel := Surface.source h
  backend := machine M R
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then row else if ax=1 then batch else 0
  numPids := some (fun ax => if ax=0 then h.N else if ax=1 then h.batches else 1)

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section

/-- The unmasked EVEN_K branch really reads BK entries, including K=0. -/
def inputWidth (h : Call) := if h.even then h.BK else h.K
def inputs (h : Call) : List IOChannel :=
  [⟨.real,[h.batches,inputWidth h],"X"⟩,
   ⟨.fp16,[h.batches,h.N,h.K],"W"⟩,
   ⟨.real,[h.batches,h.N],"Y"⟩,
   ⟨.int,[h.batches],"Indices"⟩]
abbrev Inputs (A : Type) (h : Call) := ChannelValues A (inputs h)

def index {h : Call} (xs : Inputs A h) (batch : Fin h.batches) : Int :=
  xs ⟨3,by simp [inputs]⟩ (batch,PUnit.unit)

def Valid {h : Call} (xs : Inputs A h) : Prop :=
  ∀ batch,index xs batch=-1 ∨ 0 ≤ index xs batch

def input [Inhabited A] {h : Call} (xs : Inputs A h) (batch : Fin h.batches) (k : Nat) : A :=
  if hk : k < inputWidth h then xs ⟨0,by simp [inputs]⟩ (batch,⟨k,hk⟩,PUnit.unit) else default
def weight [Inhabited A] {h : Call} (xs : Inputs A h)
    (batch : Fin h.batches) (row : Fin h.N) (k : Nat) : A :=
  if hk : k<h.K then xs ⟨1,by simp [inputs]⟩ (batch,row,⟨k,hk⟩,PUnit.unit) else default
def previous {h : Call} (xs : Inputs A h) (batch : Fin h.batches) (row : Fin h.N) : A :=
  xs ⟨2,by simp [inputs]⟩ (batch,row,PUnit.unit)

def result (h : Call) (f : Fin h.batches → Fin h.N → A) :
    ChannelValues A [⟨.real,[h.batches,h.N],"Y"⟩] :=
  Fin.cases (fun j => f j.1 j.2.1) (fun k => k.elim0)

/-- Local typed operation-order reference, independent of execution and
physical addresses. The full BK products, half reduction and FP32 optional
input-add are retained; public Real/native formulas specialize this graph. -/
def localValue (M : Algebra A) (R : Reduction A) (h : Call)
    (x w : Nat → A) (old : A) : A :=
  let tile : Values A .fp16 [256,h.BK] := fun j =>
    M.binary (some .fp32) .fp16 .mul
      (M.cast (some .fp32) .real .fp16 (M.fp32Load
        (if h.even=Bool.true ∨ j.2.1.val<h.K then x j.2.1.val else M.literal (some .fp32) .real 0)))
      (if j.1.val<1 ∧ j.2.1.val<h.K then w j.2.1.val
       else M.cast (some .fp32) .real .fp16 (M.literal (some .fp32) .real 0.0))
  let total := M.cast (some .fp32) .fp16 .real
    (R.apply (some .fp32) .fp16 ⟨1,by simp⟩ Bool.false tile (0,PUnit.unit))
  if h.add then M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .add total (M.fp32Load old)) else total

def invocation (M : Algebra A) (R : Reduction A) (h : Call) (p : Geometry.Program h) : Invocation A :=
  PublicExecution.invocation M R h p.batch p.row

/-- All batch/output-row invocations, signed skip metadata, gathered FP16
weights and the optional in-place FP32 add. Metadata has no adapter upper
bound beyond the already required backed gathered window. -/
def sourceIO (M : Algebra A) (R : Reduction A) (project : A → A) (h : Call) : ProgramIO A A where
  inputs := inputs h
  outputs := [⟨.real,[h.batches,h.N],"Y"⟩]
  scratch := []
  buffers := ["X","W","Y","Indices"]
  read := fun port xs => match port with
    | ⟨0,_⟩ => fun j => j.1.val*h.K+j.2.1.val
    | ⟨1,_⟩ => fun j => (index xs j.1).toNat*(h.N*h.K)+j.2.1.val*h.K+j.2.2.1.val
    | ⟨2,_⟩ => fun j => j.1.val*h.columns+j.2.1.val+h.slice
    | ⟨3,_⟩ => fun j => j.1.val
    | ⟨k+4,hp⟩ => False.elim (by simp [inputs] at hp)
  inputMask := fun port xs => match port with
    | ⟨0,_⟩ => fun j => 0<h.N ∧ index xs j.1≠-1
    | ⟨1,_⟩ => fun j => index xs j.1≠-1
    | ⟨2,_⟩ => fun j => index xs j.1≠-1 ∧ h.add=Bool.true
    | ⟨3,_⟩ => fun _ => 0<h.N
    | ⟨k+4,hp⟩ => False.elim (by simp [inputs] at hp)
  write := Fin.cases (fun _ j => j.1.val*h.columns+j.2.1.val+h.slice) (fun k => k.elim0)
  outputMask := Fin.cases (fun xs j => index xs j.1≠-1) (fun k => k.elim0)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (Geometry.programs h).map (invocation M R h)
  encode := id
  observe := fun _ _ => project

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PairIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
variable {A : Type} [Inhabited A]

def invocation (M : Algebra A) (R : Reduction A) (h : Call) (p : Geometry.Program h) : Invocation A :=
  { PublicIO.invocation M R h p with kernel := Kernels.outputFirst h }

def candidate (M : Algebra A) (R : Reduction A) (h : Call) :
    ProgramIO.Variant (PublicIO.sourceIO M R id h) where
  schedule := fun _ => (Geometry.programs h).map (invocation M R h)
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PairIO

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section

structure LogicalInputs (h : Call) where
  input : Fin h.batches → Fin (inputWidth h) → ℝ
  weight : Fin h.batches → Fin h.N → Fin h.K → ℝ
  previous : Fin h.batches → Fin h.N → ℝ
  indices : Fin h.batches → Int

def logicalInputsEquiv (h : Call) : Inputs ℝ h ≃ LogicalInputs h where
  toFun := fun xs => ⟨fun b k => xs (0 : Fin 4) (b,k,()),
    fun b n k => xs (1 : Fin 4) (b,n,k,()), fun b n => xs (2 : Fin 4) (b,n,()),
    fun b => xs (3 : Fin 4) (b,())⟩
  invFun := fun xs => Fin.cases (fun j => xs.input j.1 j.2.1)
    (Fin.cases (fun j => xs.weight j.1 j.2.1 j.2.2.1)
      (Fin.cases (fun j => xs.previous j.1 j.2.1)
        (Fin.cases (fun j => xs.indices j.1) (fun i => i.elim0))))
  left_inv := by
    intro vals
    funext i j
    fin_cases i
    · rcases j with ⟨b,k,⟨⟩⟩; rfl
    · rcases j with ⟨b,n,k,⟨⟩⟩; rfl
    · rcases j with ⟨b,n,⟨⟩⟩; rfl
    · rcases j with ⟨b,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputsEquiv (h : Call) :
    ChannelValues ℝ [⟨.real,[h.batches,h.N],"Y"⟩] ≃ (Fin h.batches → Fin h.N → ℝ) where
  toFun := fun vals b row => vals 0 (b,row,())
  invFun := fun f => Fin.cases (fun j => f j.1 j.2.1) (fun i => i.elim0)
  left_inv := by
    intro vals
    funext o j
    fin_cases o
    rcases j with ⟨b,row,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

def LogicalInputs.inputAt {h : Call} (xs : LogicalInputs h) (b : Fin h.batches)
    (k : Fin h.K) : ℝ := if inside : k.val < inputWidth h then xs.input b ⟨k.val,inside⟩ else 0

def logicalValid {h : Call} (xs : LogicalInputs h) : Prop :=
  ∀ b, xs.indices b = -1 ∨ 0 ≤ xs.indices b

def gemvLogicalIO (h : Call) : LogicalProgramIO ℝ ℝ (LogicalInputs h)
    (Fin h.batches → Fin h.N → ℝ) where
  core := sourceIO Real.algebra Real.reduction id h
  inputs := logicalInputsEquiv h
  outputs := logicalOutputsEquiv h

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
