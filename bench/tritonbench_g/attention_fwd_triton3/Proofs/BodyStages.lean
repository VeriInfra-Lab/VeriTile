import bench.tritonbench_g.attention_fwd_triton3.Proofs.Reference

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
variable {α : Type} [Inhabited α]

/-- Source-independent ordered score formula before the online update. -/
def scores (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α) :
    Values α .real [c.BM,c.BN] :=
  let M:=o.scalar
  let k:=loadBlock o a old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1])
  let raw:=o.dot (some .fp32) (batch:=[]) (m:=c.BM) (k:=c.D) (n:=c.BN) (query c o a) k
  let score : Values α .real [c.BM,c.BN]:=fun i=>M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) (raw i)) (scale c o)
  let windowed : Values α .real [c.BM,c.BN]:=fun i=>
    if c.window≠0 then if windowMask c a cur i then score i else M.negInf else score i
  fun i=>if c.params.evenN=0 then
    if (i.2.1.val : Int)+cur<c.NK then windowed i else M.negInf else windowed i

def nextCenter (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α) :
    Values α .real [c.BM] :=
  let reduced:=o.scalar.reduceMax (some .fp32) (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false
    (scores c o a cur old)
  fun i=>o.scalar.binary (some .fp32) .real .max (old.center i) (reduced i)

def probabilities (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α) :
    Values α .real [c.BM,c.BN] :=
  let M:=o.scalar
  let p : Values α .real [c.BM,c.BN]:=fun i=>M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (scores c o a cur old i) (nextCenter c o a cur old (i.1,())))
  let windowed : Values α .real [c.BM,c.BN]:=fun i=>
    if c.window≠0 then if windowMask c a cur i then p i else M.literal (some .fp32) .real 0 else p i
  fun i=>if c.params.evenN=0 then
    if (i.2.1.val : Int)+cur<c.NK then windowed i else M.literal (some .fp32) .real 0 else windowed i

def alphas (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α) :
    Values α .real [c.BM] := fun i=>
  let diff:=o.scalar.binary (some .fp32) .real .sub (old.center i) (nextCenter c o a cur old i)
  if !(o.eq (some .fp32) .real diff diff) then o.scalar.literal (some .fp32) .real 1
  else o.scalar.unary (some .fp32) .exp2 diff

def scaledCarry (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α) : Carry c α :=
  {old with
    mass:=fun i=>o.scalar.binary (some .fp32) .real .add
      (o.scalar.binary (some .fp32) .real .mul (old.mass i) (alphas c o a cur old i))
      (o.scalar.reduceSum (some .fp32) (shape:=[c.BM,c.BN]) ⟨1,by simp⟩ Bool.false
        (probabilities c o a cur old) i)
    weighted:=fun i=>o.scalar.binary (some .fp32) .real .mul
      (old.weighted i) (alphas c o a cur old (i.1,()))}

theorem advance_stages (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α) :
    advance c o a cur old=
      {center:=nextCenter c o a cur old
       mass:=(scaledCarry c o a cur old).mass
       weighted:=fun i=>o.scalar.binary (some .fp32) .real .add
         ((scaledCarry c o a cur old).weighted i)
         (o.dot (some .fp32) (batch:=[]) (m:=c.BM) (k:=c.BN) (n:=c.D)
           (fun j=>o.scalar.cast (some .fp32) .real .real (probabilities c o a cur old j))
           (loadBlock o a old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])) i)
       keyPointer:=old.keyPointer.advance [0,c.BN]
       valuePointer:=old.valuePointer.advance [c.BN,0]} := by rfl

def ScoresReady (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α)
    (s : State α) : Prop :=
  Matches c o a old s ∧
  s.regs .real [c.BM,c.BN] "__veritile_statement_jit_local_11_qk"=some (scores c o a cur old) ∧
  (c.window≠0 → s.regs .bool [c.BM,c.BN] "__veritile_statement_jit_local_11_mask"=
    some (windowMask c a cur)) ∧
  s.regs .int [] "__veritile_statement_jit_local_11_start_n"=some (fun _=>cur)

def UpdateReady (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α)
    (s : State α) : Prop :=
  Matches c o a (scaledCarry c o a cur old) s ∧
  s.regs .real [c.BM,c.BN] "__veritile_statement_jit_local_11_p"=
    some (probabilities c o a cur old) ∧
  s.regs .real [c.BM] "__veritile_statement_jit_local_11_m_ij"=some (nextCenter c o a cur old)

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
