import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.tritonbench_g.triton_softmax.Proofs.KernelInterface
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Memory.KernelSpec.Tile
import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Bench.Repairs.TritonSoftmax.IO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
noncomputable section

/-- The authoritative FP32 source, with arbitrary storage offsets and row/column
strides. All active input/output addresses are explicit; no private scratch. -/
def softmaxIO (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) : MaskedTileKernelIO₁ where
  kernel := softmax_kernel out inp ib ob ir ic orow oc n B
  projection := by rfl
  inp := inp
  out := out
  shape := [B]
  read := fun pid i => ib+pid*ir+i.1.val*ic
  write := fun pid i => ob+pid*orow+i.1.val*oc
  mask := fun _ i => i.1.val<n


/-- Native FP32 IO for either retained source. The source difference is only
integer input-row multiplication order. Each output observes its actual owning
thread in the fixed-four-warp graph; all FP32 input words are admitted. -/
def nativeIO (inst : NativeGraph.Instructions) (candidate : Bool) (x y : View)
    (B p : Nat) : ProgramIO NativeGraph.Word NativeGraph.Node where
  inputs := [⟨.real,[B],x.region⟩]
  outputs := [⟨.real,[B],y.region⟩]
  scratch := []
  buffers := [x.region,y.region]
  read := by intro k _; have hk : k=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance k 0; subst k; exact fun i => x.index p i.1.val
  inputMask := by intro k _; have hk : k=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance k 0; subst k; exact fun i => i.1.val<x.cols
  write := by intro k _; have hk : k=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance k 0; subst k; exact fun i => y.index p i.1.val
  outputMask := by intro k _; have hk : k=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance k 0; subst k; exact fun i => i.1.val<x.cols
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [{
    kernel := if candidate then
      indexCommutedKernel y.region x.region x.offset y.offset x.rowStride x.colStride
        y.rowStride y.colStride x.cols B
      else softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride
        y.rowStride y.colStride x.cols B
    backend := SoftmaxTriton1.Floating.backend (NativeGraph.algebra inst 1)
    pids := fun ax => if ax=0 then p else 0
    numPids := some (fun ax => if ax=0 then x.rows else 1) }]
  encode := NativeGraph.atom
  observe := by intro k; have hk : k=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance k 0; subst k; exact fun i value => value (NativeGraph.owner 1 B i.1.val)


/-- Lift an ordinary logical row function to the explicitly declared single
input/output channels. This contains no execution or correctness predicate. -/
def nativeValues (inst : NativeGraph.Instructions) (candidate : Bool) (x y : View)
    (B p : Nat) (f : (Fin B → NativeGraph.Word) → Fin B → NativeGraph.Word) :
    (nativeIO inst candidate x y B p).Inputs → (nativeIO inst candidate x y B p).Outputs := by
  intro xs k
  have hk : k=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance k 0
  subst k
  exact fun i => f (fun j => xs (0 : Fin 1) (j,PUnit.unit)) i.1


/-- The retained actual source candidate, with the identical fixed-four-warp
native observation and the same public channels, masks and windows. -/
def indexCandidate (inst : NativeGraph.Instructions) (x y : View) (B p : Nat) :
    ProgramIO.Variant (nativeIO inst Bool.false x y B p) where
  schedule := (nativeIO inst Bool.true x y B p).schedule
  observe := (nativeIO inst Bool.true x y B p).observe

def original (inst : NativeGraph.Instructions) (x y : View) (B p : Nat) :=
  ProgramIO.Implementation.original (nativeIO inst Bool.false x y B p) (fun _ => True)

def indexCommuted (inst : NativeGraph.Instructions) (x y : View) (B p : Nat) :=
  ProgramIO.Implementation.ofVariant (indexCandidate inst x y B p) (fun _ => True)

/-- All padded source values are retained. `active` selects exactly the
logical n-column row; values in inactive lanes remain arbitrary. -/
structure Row (n B : Nat) where
  values : Fin B → ℝ

def Row.active {n B : Nat} (xs : Row n B) (cover : n≤B) (j : Fin n) : ℝ :=
  xs.values (Fin.castLE cover j)

def rowInputs (n B : Nat) : (Nat × (TileIndex [B] → ℝ)) ≃ (Nat × Row n B) where
  toFun xs := (xs.1,⟨fun j => xs.2 (j,())⟩)
  invFun xs := (xs.1,fun i => xs.2.values i.1)
  left_inv := by rintro ⟨pid,xs⟩; rfl
  right_inv := by rintro ⟨pid,xs⟩; cases xs; rfl

def rowOutputs (B : Nat) : (TileIndex [B] → ℝ) ≃ (Fin B → ℝ) where
  toFun xs j := xs (j,())
  invFun xs i := xs i.1
  left_inv := by intro xs; funext i; rcases i with ⟨j,u⟩; cases u; rfl
  right_inv := by intro xs; rfl

def rowIO (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :
    LogicalIO (Nat × (TileIndex [B] → ℝ)) (TileIndex [B] → ℝ)
      (Nat × Row n B) (Fin B → ℝ)
      (fun f => (softmaxIO out inp ib ob ir ic orow oc n B).Implements
        (fun pid xs => f (pid,xs))) where
  inputs := rowInputs n B
  outputs := rowOutputs B

end
end VeriTile.Bench.Repairs.TritonSoftmax.IO
