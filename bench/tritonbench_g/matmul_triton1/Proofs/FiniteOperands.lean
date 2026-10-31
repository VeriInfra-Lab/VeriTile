import bench.tritonbench_g.matmul_triton1.Proofs.FiniteIndex
import bench.tritonbench_g.matmul_triton1.Proofs.FPComplete

/-! Both concrete kernels share this masked finite64 loop and pointer model.
The final task adapters connect it to each distinct typed program/grid mapping.
The dot primitive and every padded lane are preserved. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteOperands
open VeriTile.Triton FP.Structural
open FiniteIndex
open Core (Primitives matrixDot AIndex BIndex CIndex)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

def naturalA (P : Primitives α) (p : Parameters) (grouped : Bool)
    (s : State α) (k : Nat) (i : AIndex p) : α :=
  let r := naturalRow p grouped (s.pids 0)*p.blockRows+i.1.val
  if r < p.rows ∧ i.2.1.val < p.innerSize-k*p.blockInner then
    (s.mem p.input (r*p.innerSize+(k*p.blockInner+i.2.1.val))).read .real else P.zero

def naturalB (P : Primitives α) (p : Parameters) (grouped : Bool)
    (s : State α) (k : Nat) (i : BIndex p) : α :=
  let c := naturalColumn p grouped (s.pids 0)*p.blockColumns+i.2.1.val
  if i.1.val < p.innerSize-k*p.blockInner ∧ c < p.columns then
    (s.mem p.weight ((k*p.blockInner+i.1.val)*p.columns+c)).read .real else P.zero

def loadA (P : Primitives α) (p : Parameters) (grouped : Bool)
    (s : State α) (k : BitVec 64) (i : AIndex p) : α :=
  if (row p grouped (s.pids 0) i.1.val).ult (word p.rows) &&
      (word i.2.1.val).ult (word p.innerSize-k*word p.blockInner) then
    (s.mem p.input (aOffset p grouped (s.pids 0) i.1.val i.2.1.val k.toNat).toNat).read .real else P.zero

def loadB (P : Primitives α) (p : Parameters) (grouped : Bool)
    (s : State α) (k : BitVec 64) (i : BIndex p) : α :=
  if (word i.1.val).ult (word p.innerSize-k*word p.blockInner) &&
      (column p grouped (s.pids 0) i.2.1.val).ult (word p.columns) then
    (s.mem p.weight (bOffset p grouped (s.pids 0) i.2.1.val i.1.val k.toNat).toNat).read .real else P.zero

theorem loads (P : Primitives α) (p : Parameters) (h : Ready p) (storage : StorageBounds p)
    (grouped : Bool) (s : State α) (hp : s.pids 0 < count p) (k : Nat)
    (hk : k < ceil p.innerSize p.blockInner) :
    loadA P p grouped s (word k) = naturalA P p grouped s k ∧
    loadB P p grouped s (word k) = naturalB P p grouped s k := by
  have small : k < 2^64 := by have := loop_bound_small p h; omega
  constructor
  · funext i
    have m := masks p h grouped (s.pids 0) i.1.val 0 k i.2.1.val hp i.1.isLt
      h.blockColumns.1 hk i.2.1.isLt
    simp only [loadA,naturalA,Bool.and_eq_true,m.1,m.2.2,word_exact small]
    split
    · rename_i active
      rw [a_active p h storage grouped _ _ _ _ hp active.1 active.2]
    · rfl
  · funext i
    have m := masks p h grouped (s.pids 0) 0 i.2.1.val k i.1.val hp h.blockRows.1
      i.2.1.isLt hk i.1.isLt
    simp only [loadB,naturalB,Bool.and_eq_true,m.2.1,m.2.2,word_exact small]
    split
    · rename_i active
      rw [b_active p h storage grouped _ _ _ _ hp active.2 active.1]
    · rfl

def accumulation (P : Primitives α) (p : Parameters) (grouped : Bool)
    (s : State α) : Nat → CIndex p → α
  | 0 => fun _ => P.zero
  | k+1 => matrixDot P (batch:=[]) (naturalA P p grouped s k) (naturalB P p grouped s k)
      (accumulation P p grouped s k)

def active (p : Parameters) (grouped : Bool) (s : State α) (i : CIndex p) : Prop :=
  naturalRow p grouped (s.pids 0)*p.blockRows+i.1.val < p.rows ∧
    naturalColumn p grouped (s.pids 0)*p.blockColumns+i.2.1.val < p.columns
instance (p : Parameters) (grouped : Bool) (s : State α) (i : CIndex p) :
    Decidable (active p grouped s i) := inferInstanceAs (Decidable (_ ∧ _))

def offset (p : Parameters) (grouped : Bool) (s : State α) (i : CIndex p) : Nat :=
  (naturalRow p grouped (s.pids 0)*p.blockRows+i.1.val)*p.columns+
    (naturalColumn p grouped (s.pids 0)*p.blockColumns+i.2.1.val)

def written (P : Primitives α) (p : Parameters) (grouped : Bool) (s : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p grouped s i then t.write p.output (offset p grouped s i)
      (.mk .real (accumulation P p grouped s (ceil p.innerSize p.blockInner) i)) else t) s

#axiomsClean loads
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteOperands
