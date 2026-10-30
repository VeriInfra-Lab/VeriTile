import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.FP
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- A scalar/tile reference without kernel execution or storage. All source
casts, operand orders, inactive literals and reduction callbacks are explicit.
It applies to arbitrary words, including exceptional FP payloads. -/
def maskedValue {α : Type} (M : Algebra α) (fallback : α) (K row : Nat)
    (causal : Bool) {D : Nat} (values : Fin D → α) (j : Fin D) : α :=
  M.fp32Load (if Forward.readActive K row causal j.val then values j else fallback)

def selectedValue {α : Type} (M : Algebra α) (fallback : α) (K row : Nat)
    (causal : Bool) {D : Nat} (values : Fin D → α) (j : Fin D) : α :=
  if causal && decide (row<j.val) then M.cast (some .fp32) .real .real fallback
  else maskedValue M fallback K row causal values j

def rowSum {α : Type} (M : Algebra α) {D : Nat} (values : Fin D → α) : α :=
  M.reduceSum (some .fp32) (shape := [D]) ⟨0,by simp⟩ Bool.false (fun j => values j.1) PUnit.unit

def rowMax {α : Type} (M : Algebra α) {D : Nat} (values : Fin D → α) : α :=
  M.reduceMax (some .fp32) (shape := [D]) ⟨0,by simp⟩ Bool.false (fun j => values j.1) PUnit.unit

def forwardTail {α : Type} (M : Algebra α) (log : Bool) {D : Nat}
    (score : Fin D → α) (i : Fin D) : Cell α :=
  let z := fun j => M.binary (some .fp32) .real .sub (score j) (rowMax M score)
  let num := fun j => M.unary (some .fp32) .exp (z j)
  let den := rowSum M num
  Cell.mk .real (M.cast (some .fp32) .real .real
    (if log then M.binary (some .fp32) .real .sub (z i) (M.unary (some .fp32) .log den)
     else M.binary (some .fp32) .real .div (num i) den))

def forwardValue {α : Type} (M : Algebra α) (K row : Nat) (log causal : Bool)
    (mode : Forward.MaskMode) {D : Nat} (xs masks : Fin D → α) (i : Fin D) : Cell α :=
  let x := selectedValue M M.negInf K row causal xs
  let score := fun j => if mode.present then M.binary (some .fp32) .real .add (x j)
    (maskedValue M M.negInf K row causal masks j) else x j
  forwardTail M log score i

def backwardValue {α : Type} (M : Algebra α) (K row : Nat) (log causal : Bool)
    {D : Nat} (gs os : Fin D → α) (i : Fin D) : Cell α :=
  let zero := M.literal (some .fp32) .real 0
  let g := selectedValue M zero K row causal gs
  let o := selectedValue M zero K row causal os
  let v := if log then M.binary (some .fp32) .real .sub (g i)
      (M.binary (some .fp32) .real .mul (M.unary (some .fp32) .exp (o i)) (rowSum M g))
    else M.binary (some .fp32) .real .mul (o i)
      (M.binary (some .fp32) .real .sub (g i)
        (rowSum M (fun j => M.binary (some .fp32) .real .mul (g j) (o j))))
  Cell.mk .real (M.cast (some .fp32) .real .real
    (if causal then if i.val≤row then v else zero else v))

theorem backward_value {α : Type} [Inhabited α] (M : Algebra α)
    (Dst G O : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal : Bool) (s : State α) (i : Fin D) (hi : i.val<K) :
    (Floating.backward_run M Dst G O bm bn gm gn om on K D log causal s).val.mem Dst
      (s.pids 0*bm+s.pids 1*bn+i.val)=
      backwardValue M K (s.pids 1) log causal
        (fun j => (s.mem G (s.pids 0*gm+s.pids 1*gn+j.val)).read .real)
        (fun j => (s.mem O (s.pids 0*om+s.pids 1*on+j.val)).read .real) i := by
  cases log <;> cases causal
  all_goals
    dsimp only [Floating.backward_run]
    rw [State.masked_scatter_readback (shape := [D]) Dst
      (fun j => s.pids 0*bm+s.pids 1*bn+j.1.val) _ (fun j => j.1.val<K) _ (i,PUnit.unit) hi
      (by intro a b h; rcases a with ⟨a,⟨⟩⟩; rcases b with ⟨b,⟨⟩⟩; congr 1; apply Fin.ext; simpa using h)]
    have hz : (0.0 : ℝ)=0 := by norm_num
    simp only [backwardValue,selectedValue,maskedValue,rowSum,Forward.readActive,
      Bool.false_eq_true,Bool.false_and,Bool.true_and,
      Bool.not_false,Bool.not_true,Bool.true_or,Bool.false_or,Bool.and_true,
      Bool.and_false,decide_eq_true_eq,Bool.and_eq_true,if_false,if_true,hz,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
    try rfl

theorem forward_value {α : Type} [Inhabited α] (M : Algebra α)
    (Y X Mask : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0<D)
    (log : Bool) (mode : Forward.MaskMode) (causal : Bool) (s : State α)
    (i : Fin D) (hi : i.val<K) :
    (Floating.forward_run M Y X Mask ym yn xm xn sm K D hD log mode causal s).val.mem Y
      (s.pids 0*ym+s.pids 1*yn+i.val)=
      forwardValue M K (s.pids 1) log causal mode
        (fun j => (s.mem X (s.pids 0*xm+s.pids 1*xn+j.val)).read .real)
        (fun j => (s.mem Mask (Forward.maskAddress mode (s.pids 0) (s.pids 1) sm j.val)).read .real) i := by
  cases mode <;> cases log <;> cases causal
  all_goals
    dsimp only [Floating.forward_run,Forward.TypedTotal.fp32_source_run]
    rw [State.masked_scatter_readback (shape := [D]) Y
      (fun j => s.pids 0*ym+s.pids 1*yn+j.1.val) _ (fun j => j.1.val<K) _ (i,PUnit.unit) hi
      (by intro a b h; rcases a with ⟨a,⟨⟩⟩; rcases b with ⟨b,⟨⟩⟩; congr 1; apply Fin.ext; simpa using h)]
    simp only [forwardValue,forwardTail,selectedValue,maskedValue,rowSum,rowMax,Forward.readActive,
      Forward.MaskMode.present,Forward.maskAddress,
      Bool.false_eq_true,Bool.false_and,Bool.true_and,
      Bool.not_false,Bool.not_true,Bool.true_or,Bool.false_or,Bool.and_true,
      Bool.and_false,decide_eq_true_eq,Bool.and_eq_true,if_false,if_true,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
    try rfl

#specNonCircular forwardValue avoiding [forward,Floating.forward_run,runRows,runRow]
#specNonCircular backwardValue avoiding [backward,Floating.backward_run,runRows,runRow]
#axiomsClean backward_value
#axiomsClean forward_value
end
end VeriTile.Bench.Repairs.Ksoftmax
