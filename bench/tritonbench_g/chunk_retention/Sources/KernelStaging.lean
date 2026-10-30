import bench.tritonbench_g.chunk_retention.Sources.KernelModel
import bench.tritonbench_g.chunk_retention.Sources.Arithmetic

/-! Local expression staging for the selected IEEE-FP32 matrix profile.
An addition consuming a live dot supplies its actual accumulator to the ordered
scalar FMA chain. Casts and explicit materialization end that lifetime. This
file describes local compiler/ISA interpretation, not compiler correctness.
The source execution theorems quantify arbitrary algebras independently. -/
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging
open VeriTile Triton FP.Structural KernelModel
noncomputable section

inductive Term (α : Type) where
  | value : α → Term α
  | input : RegionName → Nat → α → Term α
  | literal : ℝ → Term α
  | integer : Int → Term α
  | binary : Binary → Term α → Term α → Term α
  | unary : Unary → Term α → Term α
  | cast : Term α → Term α
  | load : Term α → Term α
  | materialize : Term α → Term α
  | pendingDot (k : Nat) (a b : Fin k → Term α)

instance [Inhabited α] : Inhabited (Term α) := ⟨.value default⟩

/-- `literal` is the scalar constant/host-argument conversion port. The selected
source uses only add/sub/mul, exp2 and log2. Other Algebra constructors have a
total unused interpretation and are absent from the quotation audit. -/
def lower (ops : Ops α) (literal : ℝ → α) : Term α → α
  | .value x => x
  | .input _ _ x => x
  | .literal x => literal x
  | .integer n => ops.integer n
  | .binary .add a (.pendingDot k left right) =>
      chain ops k
        (fun i => if hi : i<k then lower ops literal (left ⟨i,hi⟩) else ops.zero)
        (fun i => if hi : i<k then lower ops literal (right ⟨i,hi⟩) else ops.zero)
        (lower ops literal a)
  | .binary .add a b => ops.add (lower ops literal a) (lower ops literal b)
  | .binary .sub a b => ops.sub (lower ops literal a) (lower ops literal b)
  | .binary .mul (.integer n) (.literal one) =>
      if one=1 then ops.integer n else ops.mul (ops.integer n) (literal one)
  | .binary .mul a b => ops.mul (lower ops literal a) (lower ops literal b)
  | .binary _ _ _ => ops.zero
  | .unary .exp2 a => ops.exp2 (lower ops literal a)
  | .unary .log2 a => ops.log2 (lower ops literal a)
  | .unary _ _ => ops.zero
  | .cast a => ops.cast (lower ops literal a)
  | .load a | .materialize a => lower ops literal a
  | .pendingDot k left right =>
      chain ops k
        (fun i => if hi : i<k then lower ops literal (left ⟨i,hi⟩) else ops.zero)
        (fun i => if hi : i<k then lower ops literal (right ⟨i,hi⟩) else ops.zero)
        ops.zero

def algebra (zero : α) : Algebra (Term α) where
  literal := fun _ _ x => .literal x
  negInf := .value zero
  binary := fun _ _ => .binary
  unary := fun _ => .unary
  cast := fun _ _ _ => .cast
  fromNat := fun _ n => .integer n
  fromInt := fun _ n => .integer n
  fp32Bits := fun _ => .value zero
  fp32Load := .load
  reduceSum := fun _ {_} _ _ _ _ => .value zero
  reduceMax := fun _ {_} _ _ _ _ => .value zero

def dotValues (l r out : FloatDType) :
    (batch : TileShape) → {m k n : Nat} →
      Values (Term α) l.toTileDType (batch++[m,k]) →
      Values (Term α) r.toTileDType (batch++[k,n]) →
      Values (Term α) out.toTileDType (batch++[m,n])
  | [], _, k, _, a, b, i => ofFloat out (.pendingDot k
      (fun e => toFloat l (a ⟨i.1,e,()⟩)) (fun e => toFloat r (b ⟨e,i.2.1,()⟩)))
  | _::ds, _, _, _, a, b, i => dotValues l r out ds (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

/-- Every dot in the selected source has `allow_tf32=False`; the fixed local
profile is ordered FP32 scalar FMA. There is no matrix result callback here. -/
def dot : DotPrimitive (Term α) where
  apply := fun _ l r out _ {batch} {_} {_} {_} a b => dotValues l r out batch a b

def fromInput (role : RegionName) : Term α → Bool
  | .input r _ _ => r == role
  | .load a | .cast a => fromInput role a
  | _ => Bool.false
def weightedScore : Term α → Bool
  | .cast (.binary .mul (.binary .add _ (.pendingDot _ _ _)) _) => Bool.true
  | _ => Bool.false

/-- Local operand-provenance profile for the QKV transposed result. It examines
the first reduction-lane operands of this one dot. The active-store adapter
proves that lane is present and that these tags identify the q × dS dot.
No complete kernel or output equality is supplied to this classifier. -/
def gradientDotValues (l r out : FloatDType) :
    (batch : TileShape) → {m k n : Nat} →
      Values (Term α) l.toTileDType (batch++[m,k]) →
      Values (Term α) r.toTileDType (batch++[k,n]) →
      Values (Term α) out.toTileDType (batch++[m,n])
  | [], _, k, _, a, b, i =>
      let term := Term.pendingDot k (fun e => toFloat l (a ⟨i.1,e,()⟩))
        (fun e => toFloat r (b ⟨e,i.2.1,()⟩))
      ofFloat out (if hk : 0<k then
        if fromInput "q" (toFloat l (a ⟨i.1,⟨0,hk⟩,()⟩)) &&
          weightedScore (toFloat r (b ⟨⟨0,hk⟩,i.2.1,()⟩))
        then .materialize term else term else term)
  | _::ds, _, _, _, a, b, i => gradientDotValues l r out ds
      (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2
def gradientDot : DotPrimitive (Term α) where
  apply := fun _ l r out _ {batch} {_} {_} {_} a b => gradientDotValues l r out batch a b

theorem consume_dot (ops : Ops α) (literal : ℝ→α) (k : Nat)
    (left right : Fin k→Term α) (seed : Term α) :
    lower ops literal (.binary .add seed (.pendingDot k left right))=
      chain ops k
        (fun i => if hi : i<k then lower ops literal (left ⟨i,hi⟩) else ops.zero)
        (fun i => if hi : i<k then lower ops literal (right ⟨i,hi⟩) else ops.zero)
        (lower ops literal seed) := by cases seed <;> rfl

/-- A local layout/materialization boundary explicitly ends dot fusion. The
QKV adapter must place this at the transposed dK dot result; it must not apply
the ordinary live-dot rule across that boundary. -/
theorem materialized_add (ops : Ops α) (literal : ℝ→α) (a b : Term α) :
    lower ops literal (.binary .add a (.materialize b))=
      ops.add (lower ops literal a) (lower ops literal b) := by cases a <;> rfl

structure LiteralLaws (ops : Ops α) (literal : ℝ→α) : Prop where
  zero : literal 0=ops.zero
  one : literal 1=ops.integer 1
  negativeFive : literal (-5)=ops.integer (-5)
  blockSize : literal 64=ops.integer 64

#specNonCircular LiteralLaws avoiding [lower,FP.Structural.exec]
#axiomsClean consume_dot
#axiomsClean materialized_add
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging
