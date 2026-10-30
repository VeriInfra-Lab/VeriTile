import bench.tritonbench_g.rbe_triton_transform.Kernels
import bench.tritonbench_g.rbe_triton_transform.Proofs.Scalar
import bench.tritonbench_g.rbe_triton_transform.Proofs.Runtime
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Logic.Equiv.Fin.Basic

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section

/-- One complete selected quotation, with the exact two-axis launch ABI. -/
def invocation (ops : Operations β) (r : Request) (X Y : RegionName)
    (q : Nat × Nat) : Invocation β where
  kernel := source (parameters r X Y)
  backend := ops.backend
  clearRegisters := Bool.true
  pids := fun d => if d=0 then q.1 else if d=1 then q.2 else 0

/-- Public full-grid IO. Both channels cover the entire contiguous FP32 tensor;
there is no scratch. The schedule is supplied in its existing legal order.
`encode` and `observe` expose the Real/native carrier interpretation only. -/
@[reducible] def rotationIO (ops : Operations β) (r : Request) (X Y : RegionName)
    (qs : List (Nat × Nat)) (encode : α → β) (observe : β → α) : ProgramIO α β where
  inputs := [⟨.fp32,[r.size],X⟩]
  outputs := [⟨.fp32,[r.size],Y⟩]
  scratch := []
  buffers := [X,Y]
  read := by
    intro i
    have h : i=0 := Fin.eq_zero i
    subst i
    exact fun _ j => j.1.val
  inputMask := fun _ _ _ => True
  write := by
    intro o
    have h : o=0 := Fin.eq_zero o
    subst o
    exact fun _ j => j.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => qs.map (invocation ops r X Y)
  encode := encode
  observe := fun _ _ => observe

/-- A logical input array; the default is used only outside its declared shape. -/
def logicalInput [Inhabited α] (r : Request) (xs : TileIndex [r.size] → α) (i : Nat) : α :=
  if h : i<r.size then xs (⟨i,h⟩,PUnit.unit) else default

/-- The independent dense tensor coordinates corresponding to a flat output. -/
def coordinates (r : Request) (i : Nat) : Nat × Nat × Nat :=
  (i/(r.rows*r.columns),(i/r.columns)%r.rows,i%r.columns)

/-- Package one logical FP32 output channel without any source dependency. -/
def packOutput (r : Request) (Y : RegionName) (f : TileIndex [r.size] → α) :
    ChannelValues α [⟨.fp32,[r.size],Y⟩] := by
  intro o
  have h : o=0 := Fin.eq_zero o
  subst o
  exact f

end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Pair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section

def invocation {β : Type} (ops : Operations β) (r : Request) (X Y : RegionName)
    (q : Runtime.Program) : Invocation β :=
  { RbeTritonTransformCheckedLaunch.invocation ops r X Y q with kernel := maskReuse (parameters r X Y) }

def variant {α β : Type} (ops : Operations β) (r : Request) (X Y : RegionName)
    (qs : List Runtime.Program) (encode : α → β) (observe : β → α) :
    ProgramIO.Variant (rotationIO ops r X Y qs encode observe) where
  schedule := fun _ => qs.map (invocation ops r X Y)
  observe := fun _ _ => observe

def original {α β : Type} (ops : Operations β) (r : Request) (X Y : RegionName)
    (qs : List Runtime.Program) (encode : α → β) (observe : β → α) :=
  ProgramIO.Implementation.original (rotationIO ops r X Y qs encode observe) (fun _ => True)

def reuseMasks {α β : Type} (ops : Operations β) (r : Request) (X Y : RegionName)
    (qs : List Runtime.Program) (encode : α → β) (observe : β → α) :=
  ProgramIO.Implementation.ofVariant (variant ops r X Y qs encode observe) (fun _ => True)

end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Pair

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.LogicalView
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section
abbrev Tensor (r : Request) := Fin r.batches → Fin r.rows → Fin r.columns → ℝ

def indices (r : Request) : ((Fin r.batches × Fin r.rows) × Fin r.columns) ≃ Fin r.size :=
  (Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv

def tensorEquiv (r : Request) (X : RegionName) :
    ChannelValues ℝ [⟨.fp32,[r.size],X⟩] ≃ Tensor r where
  toFun := fun xs b m c => xs 0 (indices r ((b,m),c),())
  invFun := fun xs => Fin.cases (fun j =>
    xs ((indices r).symm j.1).1.1 ((indices r).symm j.1).1.2 ((indices r).symm j.1).2)
      (fun k => k.elim0)
  left_inv := by
    intro xs; funext o j
    have ho : o=0 := Fin.eq_zero o
    subst o
    change xs 0 ((indices r) ((indices r).symm j.1),())=xs 0 j
    rw [Equiv.apply_symm_apply]
    rfl
  right_inv := by
    intro xs; funext b m c
    change xs ((indices r).symm ((indices r) ((b,m),c))).1.1
      ((indices r).symm ((indices r) ((b,m),c))).1.2
      ((indices r).symm ((indices r) ((b,m),c))).2=xs b m c
    rw [Equiv.symm_apply_apply]

/-- The absent odd partner has the mathematical value zero. -/
def column (r : Request) (xs : Tensor r) (b : Fin r.batches) (m : Fin r.rows) (c : Nat) : ℝ :=
  if hc : c<r.columns then xs b m ⟨c,hc⟩ else 0

def rotationIO (r : Request) (X Y : RegionName) (qs : List Runtime.Program) :
    LogicalProgramIO ℝ ℝ (Tensor r) (Tensor r) where
  core := RbeTritonTransformCheckedLaunch.rotationIO ideal r X Y qs id id
  inputs := tensorEquiv r X
  outputs := tensorEquiv r Y
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.LogicalView
