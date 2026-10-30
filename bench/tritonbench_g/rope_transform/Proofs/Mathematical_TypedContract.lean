import bench.tritonbench_g.rope_transform.Proofs.Mathematical_TypedProof

/-! Independent address-indexed rotary specification. Inputs are all taken
from the launch memory; each numerical operation retains its dtype. The four
stores follow source order, so even aliasing is explicit in `referenceMemory`.
Readback theorems add the source's Q/K separation where it is necessary. -/
namespace VeriTile.Bench.TritonBenchG.RopeTransform.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev Memory (α : Type) := RegionName → Nat → Cell α

def active (heads half : Nat) {H W : Nat} (i : TileIndex [H, W]) : Bool :=
  decide (i.1.val < heads) && decide (i.2.1.val < half)

def address (pid stride headDim half : Nat) (second : Bool)
    {H W : Nat} (i : TileIndex [H, W]) : Nat :=
  pid * stride + (i.1.val * headDim + i.2.1.val) + if second then half else 0

def readFloat {α : Type} [Inhabited α] (d : FloatDType)
    (mem : Memory α) (r : RegionName) (o : Nat) : α :=
  toFloat d ((mem r o).read d.toTileDType)

def paddedZero {α : Type} (M : Algebra α) (d : FloatDType) : α :=
  M.cast none .real d (M.literal none .real 0)

/-- Coefficients are precomputed inputs. There is no sin/cos real oracle. -/
def coefficient {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (r : RegionName) (stride j : Nat) : α :=
  if j < p.headDim / 2 then readFloat d s.mem r (s.pids 0 % p.length * stride + j)
  else paddedZero M d

def component {α : Type} [Inhabited α] (M : Algebra α)
    (input rotation : FloatDType) (p : Parameters) (s : State α)
    (r : RegionName) (stride heads : Nat) (second : Bool) {H W : Nat}
    (i : TileIndex [H,W]) : α :=
  M.cast none input rotation
    (if active heads (p.headDim / 2) i then
      readFloat input s.mem r (address (s.pids 0) stride p.headDim (p.headDim / 2) second i)
    else M.cast (some .fp32) .real input (M.literal (some .fp32) .real 0))

/-- Two rounded products, then one rounded addition/subtraction, then the
actual output-storage conversion. This is intentionally not a real polynomial. -/
def laneResult {α : Type} [Inhabited α] (M : Algebra α)
    (input rotation : FloatDType) (p : Parameters) (s : State α)
    (r : RegionName) (stride heads : Nat) (second : Bool) {H W : Nat}
    (i : TileIndex [H,W]) : α :=
  let c := coefficient M rotation p s p.cos p.cosStride i.2.1.val
  let sn := coefficient M rotation p s p.sin p.sinStride i.2.1.val
  let x := component M input rotation p s r stride heads second i
  let y := component M input rotation p s r stride heads (!second) i
  let op := if second == p.backward then Binary.sub else Binary.add
  M.cast none rotation input
    (M.binary none rotation op (M.binary none rotation .mul x c)
      (M.binary none rotation .mul y sn))

def scatter {α : Type} (mem : Memory α) (r : RegionName) (pid stride heads hd : Nat)
    (second : Bool) (H W : Nat) (value : TileIndex [H,W] → Cell α) : Memory α :=
  (TileShape.allIndices [H,W]).foldl (fun current i =>
    if active heads (hd / 2) i then
      fun name offset =>
        if name = r ∧ offset = address pid stride hd (hd / 2) second i then value i
        else current name offset
    else current) mem

def referenceMemory {α : Type} [Inhabited α] (M : Algebra α)
    (input rotation : FloatDType) (p : Parameters) (s : State α) : Memory α :=
  let write (mem : Memory α) (r : RegionName) (stride heads H : Nat) (second : Bool) :=
    scatter mem r (s.pids 0) stride heads p.headDim second H (p.paddedHeadDim / 2)
      (fun i => .mk input.toTileDType
        (ofFloat input (laneResult M input rotation p s r stride heads second i)));
  let q0 := write s.mem p.q p.qStride p.qHeads p.paddedQHeads Bool.false;
  let q1 := write q0 p.q p.qStride p.qHeads p.paddedQHeads Bool.true;
  let k0 := write q1 p.k p.kStride p.kHeads p.paddedKHeads Bool.false;
  write k0 p.k p.kStride p.kHeads p.paddedKHeads Bool.true

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    (execution M p s).val.mem = referenceMemory M .fp32 .fp32 p s := by
  rcases p with ⟨q,k,c,sn,qs,ks,cs,ss,len,nq,nk,hd,pq,pk,ph,back⟩
  cases back
  all_goals
    simp [execution, referenceMemory, scatter, laneResult, component,
      coefficient, paddedZero, readFloat, active, address, fold_writes_mem,
      ofFloat, toFloat, Region.cast, State.setReg,
      FloatDType.toTileDType, FloatDType.numericDType, Nat.add_assoc]

#axiomsClean execution_memory

end
end VeriTile.Bench.TritonBenchG.RopeTransform.Typed
