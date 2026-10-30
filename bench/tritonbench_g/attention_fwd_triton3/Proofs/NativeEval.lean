import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeExpr
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Raw

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
abbrev Word := BitVec 32

structure Shape (c : Call) : Prop where
  blockM : 0<c.BM
  blockN : 0<c.BN
  feature : 0<c.D

def rowIndex (c : Call) (shape : Shape c) (row : Nat) : Fin c.BM :=
  ⟨row%c.BM,Nat.mod_lt _ shape.blockM⟩
def keyIndex (c : Call) (shape : Shape c) (key : Nat) : Fin c.BN :=
  ⟨key%c.BN,Nat.mod_lt _ shape.blockN⟩
def dimIndex (c : Call) (shape : Shape c) (dim : Nat) : Fin c.D :=
  ⟨dim%c.D,Nat.mod_lt _ shape.feature⟩

/-- These are the only ordinary floating literals in the quotation. Runtime
scale has its own constructor and never passes through Real projection. -/
def literalWord (x : ℝ) : Word :=
  if x=0 then 0 else if x=1 then 0x3f800000
  else if x=1.4426950408889634 then 0x3fb8aa3b else 0

@[simp] theorem literal_zero : literalWord 0=0 := by simp [literalWord]
@[simp] theorem literal_one : literalWord 1=0x3f800000 := by norm_num [literalWord]

/-- Evaluate only scalar ports, exact raw-bit NaN classification, explicit
ordered trees, and consecutive fixed-K8 TF32 instruction chains. The schedule
contains shapes/owners/trees; it supplies no numerical dot or reduction law. -/
def evaluate (c : Call) (shape : Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (scale : Word) : Expr → Nat → Nat → Nat → Word
  | .input _ word,_,_,_ | .bits word,_,_,_ => word
  | .literal x,_,_,_ => literalWord x
  | .scale,_,_,_ => scale
  | .negInf,_,_,_ | .initialCenter,_,_,_ => 0xff800000
  | .center a b,row,dim,owner => ops.max
      (evaluate c shape schedule ops scale a row dim owner)
      (evaluate c shape schedule ops scale b row dim owner)
  | .add a b,row,dim,owner => ops.add
      (evaluate c shape schedule ops scale a row dim owner)
      (evaluate c shape schedule ops scale b row dim owner)
  | .product a b,row,dim,owner =>
      let x := evaluate c shape schedule ops scale a row dim owner
      let y := evaluate c shape schedule ops scale b row dim owner
      match a with
      | .queryDot .. => ops.mul y x
      | _ => ops.mul x y
  | .difference (.product a b) center,row,dim,owner =>
      let x := evaluate c shape schedule ops scale a row dim owner
      let y := evaluate c shape schedule ops scale b row dim owner
      let m := evaluate c shape schedule ops scale center row dim owner
      if schedule.fuseScoreSubtract then ops.fma y x (m ^^^ 0x80000000)
      else ops.sub (ops.mul y x) m
  | .difference a b,row,dim,owner | .centerDifference a b,row,dim,owner => ops.sub
      (evaluate c shape schedule ops scale a row dim owner)
      (evaluate c shape schedule ops scale b row dim owner)
  | .divide a b,row,dim,owner => ops.div
      (evaluate c shape schedule ops scale a row dim owner)
      (evaluate c shape schedule ops scale b row dim
        (schedule.normalizationOwner (rowIndex c shape row) (dimIndex c shape dim) owner))
  | .exponential a,row,dim,owner => ops.exp2 (evaluate c shape schedule ops scale a row dim owner)
  | .logarithm a,row,dim,owner => ops.log2 (evaluate c shape schedule ops scale a row dim owner)
  | .alpha previous center,row,dim,owner =>
      let delta := ops.sub (evaluate c shape schedule ops scale previous row dim owner)
        (evaluate c shape schedule ops scale center row dim owner)
      if Raw.isNaN delta then 0x3f800000 else ops.exp2 delta
  | .fma a b acc,row,dim,owner => ops.fma
      (evaluate c shape schedule ops scale a row dim owner)
      (evaluate c shape schedule ops scale b row dim owner)
      (evaluate c shape schedule ops scale acc row dim owner)
  | .reduction maximum row n values,_,_,owner =>
      let r := rowIndex c shape row
      if maximum then Raw.reduce ops.max
        (fun j : Fin c.BN => if h:j.val<n then
          evaluate c shape schedule ops scale (values ⟨j.val,h⟩) row j.val owner else 0)
        (schedule.maxOrder r owner).tree
      else Raw.reduce ops.add
        (fun j : Fin c.BN => if h:j.val<n then
          evaluate c shape schedule ops scale (values ⟨j.val,h⟩) row j.val (schedule.probabilityOwner r j) else 0)
        (schedule.sumOrder r owner).tree
  | .pendingDot row col k a b,_,_,owner => Raw.dot ops schedule.qk row col k
      (fun j => evaluate c shape schedule ops scale (a j) row j.val owner)
      (fun j => evaluate c shape schedule ops scale (b j) j.val col owner) 0
  | .queryDot row col k a b seed,_,_,owner => Raw.dot ops schedule.qk row col k
      (fun j => evaluate c shape schedule ops scale (a j) row j.val owner)
      (fun j => evaluate c shape schedule ops scale (b j) j.val col owner)
      (evaluate c shape schedule ops scale seed row col owner)
  | .valueDot row col k a b previous alpha,_,_,owner =>
      let r := rowIndex c shape row
      let d := dimIndex c shape col
      Raw.dot ops schedule.pv row col k
        (fun j => evaluate c shape schedule ops scale (a j) row j.val
          (schedule.pvProbabilityOwner r (keyIndex c shape j.val) owner))
        (fun j => evaluate c shape schedule ops scale (b j) j.val col owner)
        (ops.mul (evaluate c shape schedule ops scale previous row col owner)
          (evaluate c shape schedule ops scale alpha row col (schedule.accumulatorAlphaOwner r d owner)))
  | .unsupported,_,_,_ => 0

theorem scale_bits (c : Call) (shape : Shape c) (schedule : Raw.Schedule c) (ops : Raw.Ops)
    (word : Word) (row dim owner : Nat) : evaluate c shape schedule ops word .scale row dim owner=word := rfl

theorem ordered_reduction (c : Call) (shape : Shape c) (schedule : Raw.Schedule c)
    (row : Nat) (owner : Nat) :
    (schedule.maxOrder (rowIndex c shape row) owner).tree.indices.Perm (List.finRange c.BN) ∧
    (schedule.sumOrder (rowIndex c shape row) owner).tree.indices.Perm (List.finRange c.BN) :=
  Raw.schedule_covers c schedule _ owner

theorem local_equiv (c : Call) (shape : Shape c) (schedule : Raw.Schedule c)
    (actual reference : Raw.Ops) (laws : Raw.LocalLaws actual reference) (scale : Word)
    (e : Expr) (row dim owner : Nat) :
    evaluate c shape schedule actual scale e row dim owner=
      evaluate c shape schedule reference scale e row dim owner := by
  rw [Raw.operations_eq actual reference laws]

#axiomsClean scale_bits
#axiomsClean ordered_reduction
#axiomsClean local_equiv
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
