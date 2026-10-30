import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.KernelAlgebra
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def sequenceLength (s : Shape) (u : State α) : Int :=
  if s.varlen then
    (u.mem "cu" (u.pids 1+1)).read .int - (u.mem "cu" (u.pids 1)).read .int
  else s.maximum

def shift (s : Shape) (u : State α) : Int :=
  if s.offsetTensor then (u.mem "offsets" (u.pids 1)).read .int else s.scalarOffset

def dataBase (s : Shape) (u : State α) : Nat :=
  if s.varlen then
    ((u.mem "cu" (u.pids 1)).read .int * (s.heads*s.width : Nat) +
      (u.pids 2*s.width : Nat)).toNat
  else u.pids 1*(s.maximum*s.heads*s.width)+u.pids 2*s.width

def row (s : Shape) (u : State α) (lane : Nat) : Nat := u.pids 0*s.blockM+lane

def dataOffset (s : Shape) (u : State α) (lane col : Nat) : Nat :=
  dataBase s u + (row s u lane*(s.heads*s.width)+col)

def coefficientOffset (s : Shape) (u : State α) (lane pair : Nat) : Nat :=
  (((row s u lane : Int)+shift s u) * (s.half : Int)+(pair : Int)).toNat

def active (s : Shape) (u : State α) (lane col : Nat) (interleaved : Bool) : Prop :=
  (row s u lane : Int) < sequenceLength s u ∧
    col < (if interleaved then 2*s.half else s.half)
instance (s : Shape) (u : State α) (lane col : Nat) (interleaved : Bool) :
    Decidable (active s u lane col interleaved) := inferInstanceAs (Decidable (_ ∧ _))

def hasCoefficient (s : Shape) (u : State α) (lane pair : Nat) : Prop :=
  (row s u lane : Int)+shift s u < s.coefficientRows ∧ pair<s.half
instance (s : Shape) (u : State α) (lane pair : Nat) :
    Decidable (hasCoefficient s u lane pair) := inferInstanceAs (Decidable (_ ∧ _))

def cosine (s : Shape) (p : Ops α) (u : State α) (lane pair : Nat) : α :=
  if hasCoefficient s u lane pair then
    (u.mem "cos" (coefficientOffset s u lane pair)).read .real else p.one

def sine (s : Shape) (p : Ops α) (u : State α) (lane pair : Nat) : α :=
  let value := if hasCoefficient s u lane pair then
    (u.mem "sin" (coefficientOffset s u lane pair)).read .real else p.zero
  if s.conjugate then p.neg value else value

def splitValue (s : Shape) (p : Ops α) (u : State α) (lane pair : Nat) (second : Bool) : α :=
  let a := if active s u lane pair Bool.false then
    (u.mem "x" (dataOffset s u lane pair)).read .real else p.zero
  let b := if active s u lane pair Bool.false then
    (u.mem "x" (dataOffset s u lane pair+s.half)).read .real else p.zero
  let co := cosine s p u lane pair
  let sn := sine s p u lane pair
  if second then p.add (p.mul a sn) (p.mul b co) else p.sub (p.mul a co) (p.mul b sn)

def swapped (col : Nat) := col+((col+1)%2)*2-1

def interleavedValue (s : Shape) (p : Ops α) (u : State α) (lane col : Nat) : α :=
  let a := if active s u lane col Bool.true then
    (u.mem "x" (dataOffset s u lane col)).read .real else p.zero
  let b := if active s u lane (swapped col) Bool.true then
    (u.mem "x" (dataOffset s u lane (swapped col))).read .real else p.zero
  let co := cosine s p u lane (col/2)
  let sn := sine s p u lane (col/2)
  if col%2=0 then p.sub (p.mul a co) (p.mul b sn) else p.add (p.mul a co) (p.mul b sn)

/-- All value expressions use the pre-store state. This preserves the source's
load-before-store behavior even when the output is the original input Tensor. -/
def scatter (s : Shape) (u current : State α) (columns : Nat) (interleaved : Bool)
    (displacement : Nat) (value : Nat → Nat → α) : State α :=
  (TileShape.allIndices [s.blockM,columns]).foldl (fun state index =>
    if active s u index.1.val index.2.1.val interleaved then
      state.write (KernelAlgebra.regionName s.output)
        (dataOffset s u index.1.val index.2.1.val+displacement)
        (.mk .real (value index.1.val index.2.1.val))
    else state) current

/-- A matrix-store normal form used to connect the quotation to the existing
independent rotation formula and schedule proofs. It does not invoke exec. -/
def result (s : Shape) (p : Ops α) (u : State α) : State α :=
  if ((u.pids 0*s.blockM : Nat) : Int) ≥ sequenceLength s u then u
  else if s.interleaved then
    scatter s u u s.blockK Bool.true 0 (interleavedValue s p u)
  else
    let low := scatter s u u (s.blockK/2) Bool.false 0 (fun lane pair => splitValue s p u lane pair Bool.false)
    scatter s u low (s.blockK/2) Bool.false s.half (fun lane pair => splitValue s p u lane pair Bool.true)

end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
