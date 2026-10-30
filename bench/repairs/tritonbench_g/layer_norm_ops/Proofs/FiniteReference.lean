import bench.repairs.tritonbench_g.layer_norm_ops.IO
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReference

/-! Source-ordered numerical targets on logical input arrays. These definitions
have no execution state, source invocation, memory construction or runner. -/
namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]

def forwardLoad (A : Algebra α) (p : ForwardParameters)
    (x : TileIndex [p.block] → α) (i : TileIndex [p.block]) : α :=
  Forward.cast A (if i.1.val<p.columns then x i else Forward.padding A)
def forwardZ (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (i : TileIndex [p.block]) : α :=
  let x := forwardLoad A p (forwardX v) i
  if p.hasResidual then Forward.binary A .add x (forwardLoad A p (forwardResidual v) i) else x
def forwardMean (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p) : α :=
  Forward.average A p (forwardZ A p v)
def forwardCentered (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (i : TileIndex [p.block]) : α :=
  if p.rms then forwardZ A p v i else Forward.binary A .sub (forwardZ A p v i) (forwardMean A p v)
def forwardInverse (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p) : α :=
  let variance := Forward.average A p (fun i =>
    let centered := if i.1.val<p.columns then forwardCentered A p v i else Forward.literal A 0
    Forward.binary A .mul centered centered)
  Forward.binary A .div (Forward.literal A 1)
    (A.unary (some .fp32) .sqrt (Forward.binary A .add variance (Forward.literal A p.epsilon)))
def forwardY (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (i : TileIndex [p.block]) : α :=
  let base := if p.rms then forwardZ A p v i else
    Forward.unhintedBinary A .sub (forwardZ A p v i) (forwardMean A p v)
  let normalized := Forward.unhintedBinary A .mul base (forwardInverse A p v)
  let weighted := Forward.unhintedBinary A .mul normalized (Forward.cast A (forwardWeight v i))
  if p.hasBias then Forward.unhintedBinary A .add weighted (Forward.cast A (forwardBias v i)) else weighted
def forwardGraph (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p) :
    ChannelValues α (forwardOutputs p) :=
  forwardResult (fun i => A.cast none .real .fp32 (forwardY A p v i))
    (fun i => A.cast none .real .fp32 (forwardZ A p v i))
    (A.cast none .real .fp32 (forwardMean A p v))
    (A.cast none .real .fp32 (forwardInverse A p v))

def extend {β : Type} [Inhabited β] {n : Nat} (f : Fin n → β) (i : Nat) : β :=
  if h : i<n then f ⟨i,h⟩ else default
@[simp] theorem extend_fin {β : Type} [Inhabited β] {n : Nat}
    (f : Fin n → β) (i : Fin n) : extend f i.val=f i := by simp [extend]

def backwardX {p : BackwardParameters} (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  extend (fun r => v ⟨0,by simp [backwardInputs]⟩ (r,i)) row
def backwardWeight {p : BackwardParameters} (v : BackwardValues α p)
    (i : TileIndex [p.block]) : α := v ⟨1,by simp [backwardInputs]⟩ i
def backwardBias {p : BackwardParameters} (v : BackwardValues α p)
    (i : TileIndex [p.block]) : α := v ⟨2,by simp [backwardInputs]⟩ i
def backwardDY {p : BackwardParameters} (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  extend (fun r => v ⟨3,by simp [backwardInputs]⟩ (r,i)) row
def backwardDres {p : BackwardParameters} (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  extend (fun r => v ⟨4,by simp [backwardInputs]⟩ (r,i)) row
def backwardMean {p : BackwardParameters} (v : BackwardValues α p) (row : Nat) : α :=
  extend (fun r => v ⟨5,by simp [backwardInputs]⟩ (r,())) row
def backwardInverse {p : BackwardParameters} (v : BackwardValues α p) (row : Nat) : α :=
  extend (fun r => v ⟨6,by simp [backwardInputs]⟩ (r,())) row

def backwardLoad (A : Algebra α) (p : BackwardParameters)
    (x : TileIndex [p.block] → α) (i : TileIndex [p.block]) : α :=
  Forward.cast A (if i.1.val<p.columns then x i else Forward.padding A)
def backwardHat (A : Algebra α) (p : BackwardParameters) (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  if i.1.val<p.columns then
    let x := backwardLoad A p (backwardX v row) i
    Forward.unhintedBinary A .mul
      (if p.rms then x else Forward.unhintedBinary A .sub x
        (A.cast none .fp32 .real (backwardMean v row)))
      (A.cast none .fp32 .real (backwardInverse v row))
  else A.literal none .real 0
def backwardWdy (A : Algebra α) (p : BackwardParameters) (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  Forward.binary A .mul (backwardLoad A p (backwardWeight v) i)
    (backwardLoad A p (backwardDY v row) i)
def backwardDX (A : Algebra α) (p : BackwardParameters) (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  let c1 := Backward.average A p (fun j => Forward.binary A .mul
    (backwardHat A p v row j) (backwardWdy A p v row j))
  let c2 := Backward.average A p (backwardWdy A p v row)
  let projection := Forward.binary A .mul (backwardHat A p v row i) c1
  let dx := Forward.binary A .mul
    (Forward.binary A .sub (backwardWdy A p v row i)
      (if p.rms then projection else Forward.binary A .add projection c2))
    (Forward.cast A (backwardInverse v row))
  if p.hasDres then Forward.binary A .add dx (backwardLoad A p (backwardDres v row) i) else dx
def backwardY (A : Algebra α) (p : BackwardParameters) (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  let y := Forward.unhintedBinary A .mul (backwardHat A p v row i)
    (backwardLoad A p (backwardWeight v) i)
  if p.hasBias then Forward.unhintedBinary A .add y (backwardLoad A p (backwardBias v) i) else y
def backwardRowDW (A : Algebra α) (p : BackwardParameters) (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  Forward.binary A .mul (backwardLoad A p (backwardDY v row) i) (backwardHat A p v row i)
def backwardRowDB (A : Algebra α) (p : BackwardParameters) (v : BackwardValues α p)
    (row : Nat) (i : TileIndex [p.block]) : α := backwardLoad A p (backwardDY v row) i
def backwardAddRow (A : Algebra α) (p : BackwardParameters) (v : BackwardValues α p)
    (row : Nat) (carry : KernelReference.Accum α p.block) : KernelReference.Accum α p.block :=
  (fun i => Forward.binary A .add (carry.1 i) (backwardRowDW A p v row i),
    if p.hasBias then fun i => Forward.binary A .add (carry.2 i) (backwardRowDB A p v row i) else carry.2)
def backwardPartials (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) : KernelReference.Accum α p.block :=
  foldRange (backwardAddRow A p v) (rowStart p pid) (rowStop p pid) 1
    (fun _ => Forward.literal A 0,fun _ => Forward.literal A 0)
def backwardResult {p : BackwardParameters}
    (dx y : Nat → TileIndex [p.block] → α) (dw db : TileIndex [p.block] → α) :
    ChannelValues α (backwardOutputs p)
  | ⟨0,_⟩ | ⟨2,_⟩ => fun j => dx j.1.val j.2
  | ⟨1,_⟩ => fun j => y j.1.val j.2
  | ⟨3,_⟩ => dw
  | ⟨4,_⟩ => db
  | ⟨i+5,h⟩ => False.elim (by simp [backwardOutputs] at h)
def backwardGraph (A : Algebra α) (p : BackwardParameters) (pid : Nat) (v : BackwardValues α p) :
    ChannelValues α (backwardOutputs p) :=
  backwardResult (fun row i => A.cast none .real .fp32 (backwardDX A p v row i))
    (fun row i => A.cast none .real .fp32 (backwardY A p v row i))
    (fun i => A.cast none .real .fp32 ((backwardPartials A p pid v).1 i))
    (fun i => A.cast none .real .fp32 ((backwardPartials A p pid v).2 i))

#specNonCircular forwardGraph avoiding [forwardKernel,FP.Structural.exec,Forward.reference]
#specNonCircular backwardGraph avoiding [backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
