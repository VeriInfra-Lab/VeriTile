import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.GridLaunch
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostBlock
import VeriTile.Triton.Memory.AlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

/-! Complete ordinary-grid Real IO. Each of the five logical input roles has
its own declared strides and base in the existing common regional memory.
The source is the already proved relocation of the shared typed kernel; no
second handwritten implementation is introduced. -/
namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
open VeriTile Triton AttnFwdTriton Verified
noncomputable section
open scoped Classical

abbrev Coordinate (p : Parameters) := OutputLayout.Coordinate p
def cells (p : Parameters) := Fintype.card (Coordinate p)
def coordinates (p : Parameters) : Coordinate p ≃ Fin (cells p) := Fintype.equivFin _
def queryScales (p : Parameters) := p.batches*p.heads*((p.context+127)/128)
def keyScales (p : Parameters) := p.batches*p.heads*((p.context+63)/64)

def arity (p : Parameters) (i : Fin 5) : Nat :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => cells p
  | ⟨3,_⟩ => queryScales p
  | ⟨4,_⟩ => keyScales p
abbrev Inputs (p : Parameters) := (i : Fin 5) → Fin (arity p i) → ℝ
abbrev Outputs (p : Parameters) := Fin 1 → Fin (cells p) → ℝ

def inputRegion (p : Parameters) (i : Fin 5) : RegionName :=
  match i with
  | ⟨0,_⟩ => p.Q | ⟨1,_⟩ => p.K | ⟨2,_⟩ => p.V
  | ⟨3,_⟩ => p.QS | ⟨4,_⟩ => p.KS

def offset (p : Parameters) (i : Fin 5) : Fin (arity p i) → Nat :=
  match i with
  | ⟨0,_⟩ => fun j => let c := (coordinates p).symm j
      RealProofs.address p.queryBatch p.queryHead p.queryRow p.queryDim
        c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val
  | ⟨1,_⟩ => fun j => let c := (coordinates p).symm j
      RealProofs.address p.keyBatch p.keyHead p.keyRow p.keyDim
        c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val
  | ⟨2,_⟩ => fun j => let c := (coordinates p).symm j
      RealProofs.address p.valueBatch p.valueHead p.valueRow p.valueDim
        c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val
  | ⟨3,_⟩ | ⟨4,_⟩ => fun j => j.val

/-- The exact flattened Algorithm source used by the original whole-grid
proof, embedded through the shared AST constructor without changing its body. -/
def source (A : FlatAlloc) (p : Parameters) : ComputeKernel :=
  let k := GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

@[simp] theorem source_projection (A : FlatAlloc) (p : Parameters) :
    (source A p).toAlgKernel=GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false := by
  simp [source,ComputeKernel.toAlgKernel,GridLaunch.kernel]

def selectedSource (A : FlatAlloc) (p : Parameters) : AlgorithmSource :=
  ⟨source A p,by simp [source,ComputeKernel.toAlgKernel]⟩

def programIO (A : FlatAlloc) (p : Parameters) : AlgorithmProgramIO where
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := [A.flat]
  ity := fun _ => .float
  iarity := arity p
  ibuf := fun _ => A.flat
  oty := fun _ => .float
  oarity := fun _ => cells p
  obuf := fun _ => A.flat
  obuf_mem := fun _ => List.mem_singleton_self _
  sarity := fun i => Fin.elim0 i
  sbuf := fun i => Fin.elim0 i
  iwin := fun i _ _ _ _ j => offset p i j + A.base (inputRegion p i)
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => OutputLayout.address p ((coordinates p).symm j)+A.base p.Out
  omask := fun _ _ _ _ _ _ => True
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i
  initialUndef := .preserve
  stages := fun _ => [.ordinary (selectedSource A p) (GridCoordinates.grid p)]

def tensor (p : Parameters) (x : Fin (cells p) → ℝ) (z h n d : Nat) : ℝ :=
  if valid : z<p.batches ∧ h<p.heads ∧ n<p.context ∧ d<p.headDim then
    x (coordinates p (⟨z,valid.1⟩,⟨h,valid.2.1⟩,⟨n,valid.2.2.1⟩,⟨d,valid.2.2.2⟩))
  else 0
def scale (x : Fin size → ℝ) (n : Nat) : ℝ := if h : n<size then x ⟨n,h⟩ else 0

def logicalInputs (p : Parameters) (values : Inputs p) : RealProofs.Inputs where
  query := tensor p (values 0)
  key := tensor p (values 1)
  value := tensor p (values 2)
  queryScale := scale (values 3)
  keyScale := scale (values 4)

/-- Independent causal softmax-weighted value, on the declared logical arrays.
Real is the mathematical interpretation, not a claim about native rounding. -/
def formula (p : Parameters) (values : Inputs p) : Outputs p := fun _ j =>
  let c := (coordinates p).symm j
  RealProofs.expected p.heads p.context p.headDim c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val
    (logicalInputs p values)

#specNonCircular formula avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
open VeriTile Triton AttnFwdTriton Verified
noncomputable section

def scaleIndex (batches heads blocks : Nat) :
    (Fin batches × Fin heads × Fin blocks) ≃ Fin (batches*heads*blocks) :=
  (Equiv.prodAssoc _ _ _).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv)

def scaleAt (values : Fin batches → Fin heads → Fin blocks → ℝ)
    (batch head block : Nat) : ℝ :=
  if valid : batch < batches ∧ head < heads ∧ block < blocks then
    values ⟨batch,valid.1⟩ ⟨head,valid.2.1⟩ ⟨block,valid.2.2⟩ else 0

structure LogicalInputs (p : Parameters) where
  query : Coordinate p → ℝ
  key : Coordinate p → ℝ
  value : Coordinate p → ℝ
  queryScale : Fin p.batches → Fin p.heads → Fin ((p.context+127)/128) → ℝ
  keyScale : Fin p.batches → Fin p.heads → Fin ((p.context+63)/64) → ℝ

def logicalInputsEquiv (p : Parameters) : Inputs p ≃ LogicalInputs p where
  toFun := fun vals => ⟨fun c => vals 0 (coordinates p c),
    fun c => vals 1 (coordinates p c), fun c => vals 2 (coordinates p c),
    fun b h block => vals 3 (scaleIndex p.batches p.heads ((p.context+127)/128) (b,h,block)),
    fun b h block => vals 4 (scaleIndex p.batches p.heads ((p.context+63)/64) (b,h,block))⟩
  invFun := fun xs => Fin.cases (fun j => xs.query ((coordinates p).symm j))
    (Fin.cases (fun j => xs.key ((coordinates p).symm j))
      (Fin.cases (fun j => xs.value ((coordinates p).symm j))
        (Fin.cases (fun j => let c := (scaleIndex p.batches p.heads ((p.context+127)/128)).symm j
          xs.queryScale c.1 c.2.1 c.2.2)
          (Fin.cases (fun j => let c := (scaleIndex p.batches p.heads ((p.context+63)/64)).symm j
            xs.keyScale c.1 c.2.1 c.2.2) (fun i => i.elim0)))))
  left_inv := by
    intro vals
    funext channel j
    fin_cases channel
    · exact congrArg (vals 0) ((coordinates p).apply_symm_apply j)
    · exact congrArg (vals 1) ((coordinates p).apply_symm_apply j)
    · exact congrArg (vals 2) ((coordinates p).apply_symm_apply j)
    · exact congrArg (vals 3) ((scaleIndex p.batches p.heads ((p.context+127)/128)).apply_symm_apply j)
    · exact congrArg (vals 4) ((scaleIndex p.batches p.heads ((p.context+63)/64)).apply_symm_apply j)
  right_inv := by
    intro xs
    cases xs with
    | mk query key value queryScale keyScale =>
      change LogicalInputs.mk (fun c => query ((coordinates p).symm (coordinates p c)))
        (fun c => key ((coordinates p).symm (coordinates p c)))
        (fun c => value ((coordinates p).symm (coordinates p c)))
        (fun b h block => queryScale
          ((scaleIndex p.batches p.heads ((p.context+127)/128)).symm
            (scaleIndex p.batches p.heads ((p.context+127)/128) (b,h,block))).1
          ((scaleIndex p.batches p.heads ((p.context+127)/128)).symm
            (scaleIndex p.batches p.heads ((p.context+127)/128) (b,h,block))).2.1
          ((scaleIndex p.batches p.heads ((p.context+127)/128)).symm
            (scaleIndex p.batches p.heads ((p.context+127)/128) (b,h,block))).2.2)
        (fun b h block => keyScale
          ((scaleIndex p.batches p.heads ((p.context+63)/64)).symm
            (scaleIndex p.batches p.heads ((p.context+63)/64) (b,h,block))).1
          ((scaleIndex p.batches p.heads ((p.context+63)/64)).symm
            (scaleIndex p.batches p.heads ((p.context+63)/64) (b,h,block))).2.1
          ((scaleIndex p.batches p.heads ((p.context+63)/64)).symm
            (scaleIndex p.batches p.heads ((p.context+63)/64) (b,h,block))).2.2) = _
      simp only [Equiv.symm_apply_apply]

def logicalOutputsEquiv (p : Parameters) : Outputs p ≃ (Coordinate p → ℝ) where
  toFun := fun vals c => vals 0 (coordinates p c)
  invFun := fun f _ j => f ((coordinates p).symm j)
  left_inv := by
    intro vals
    funext channel j
    have zero : channel=0 := Fin.eq_zero channel
    subst channel
    exact congrArg (vals 0) ((coordinates p).apply_symm_apply j)
  right_inv := by
    intro f
    funext c
    exact congrArg f ((coordinates p).symm_apply_apply c)

/-- Bounded logical tensor roles with the original zero extension for the
independent mathematical formula; arbitrary input-role aliases are retained. -/
def LogicalInputs.tensors {p : Parameters} (xs : LogicalInputs p) : RealProofs.Inputs :=
  logicalInputs p ((logicalInputsEquiv p).symm xs)

def LogicalInputs.queryScaleAt {p : Parameters} (xs : LogicalInputs p) (batch head block : Nat) : ℝ :=
  scaleAt xs.queryScale batch head block
def LogicalInputs.keyScaleAt {p : Parameters} (xs : LogicalInputs p) (batch head block : Nat) : ℝ :=
  scaleAt xs.keyScale batch head block

def attentionIO (A : FlatAlloc) (p : Parameters) :
    LogicalIO (programIO A p).Ctx (Outputs p) (LogicalInputs p) (Coordinate p → ℝ)
      (programIO A p).Implements where
  inputs := logicalInputsEquiv p
  outputs := logicalOutputsEquiv p

end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
