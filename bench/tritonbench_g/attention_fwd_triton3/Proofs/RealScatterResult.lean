import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealClosedLoop
import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeFrame

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def laneValue (c : Call) (carry : RealLoopFormula.Carry c.BM c.D) : Lane c.BM c.D → R
  | .center r=>if c.finish≠0 then
      carry.center r |>.map (fun m=>m+Real.log (carry.denominator r)/Real.log 2)
    else carry.center r
  | .denominator r=>some (carry.denominator r)
  | .output r d=>some (if c.finish≠0 then carry.numerator (r,d,())/carry.denominator r else carry.numerator (r,d,()))

theorem finalCenter_encoded (c : Call) (old : Reference.Carry c R)
    (carry : RealLoopFormula.Carry c.BM c.D) (encoded : Encodes c old carry) (r : Fin c.BM) :
    Reference.finalCenter c ops old (r,())=laneValue c carry (.center r) := by
  unfold Reference.finalCenter
  rw [encoded.1,encoded.2.1]
  by_cases hf : c.finish≠0 <;> simp [laneValue,hf,ops,algebra]
  cases carry.center r <;> rfl

theorem finalOutput_encoded (c : Call) (old : Reference.Carry c R)
    (carry : RealLoopFormula.Carry c.BM c.D) (encoded : Encodes c old carry) (r : Fin c.BM) (d : Fin c.D) :
    Reference.finalOutput c ops old (r,d,())=laneValue c carry (.output r d) := by
  unfold Reference.finalOutput
  rw [encoded.2.1,encoded.2.2]
  by_cases hf : c.finish≠0 <;> simp [laneValue,hf,ops,algebra]

theorem prepared_coordinate (c : Call) (s : State R) (idx : GridIndex c.grid) :
    (Reference.inputs (Reference.Composition.prepare c.grid idx s)).tile=c.tile idx ∧
    (Reference.inputs (Reference.Composition.prepare c.grid idx s)).hz=c.hz idx := by
  simp [Reference.inputs,Reference.Composition.prepare,GridIndex.toPids,Call.grid,Grid.rank,Call.tile,Call.hz]

theorem scatter_result (c : Call) (valid : Valid c) (s : State R) (idx : GridIndex c.grid)
    (old : Reference.Carry c R) (carry : RealLoopFormula.Carry c.BM c.D)
    (encoded : Encodes c old carry) (lane : Lane c.BM c.D) (live : Ownership.active c lane) :
    (Reference.scatter c ops (Reference.inputs (Reference.Composition.prepare c.grid idx s)) old
      (Reference.Composition.prepare c.grid idx s)).mem (address c idx lane).1 (address c idx lane).2=
      .mk .real (laneValue c carry lane) := by
  let a:=Reference.inputs (Reference.Composition.prepare c.grid idx s)
  have coord:=prepared_coordinate c s idx
  have carryEq (r : Fin c.BM) : Reference.carryAddress c a r.val=c.carryAddress idx r.val := by
    simp [a,Reference.carryAddress,Reference.row,Call.carryAddress,Call.row,coord]
  have outEq (r : Fin c.BM) (d : Fin c.D) :
      (Reference.optr c a).address [r.val,d.val]=c.outAddress idx r.val d.val := by
    simp [Reference.optr,Reference.batch,Reference.head,Call.outAddress,Call.row,Geometry.Strides.at,coord,a,Nat.add_assoc]
  have rowInj : Function.Injective (fun i : TileIndex [c.BM]=>Reference.carryAddress c a i.1.val) := by
    rintro ⟨r,⟨⟩⟩ ⟨t,⟨⟩⟩ eq
    have h:=(carry_injective c valid idx idx r t ((carryEq r).symm.trans (eq.trans (carryEq t)))).2
    subst t
    rfl
  have outInj : Function.Injective (fun i : TileIndex [c.BM,c.D]=>(Reference.optr c a).address (TileShape.indexToList _ i)) := by
    rintro ⟨r,d,⟨⟩⟩ ⟨t,e,⟨⟩⟩ eq
    have h:=output_injective c valid idx idx r t d e ((outEq r d).symm.trans (eq.trans (outEq t e)))
    rcases h with ⟨_,rfl,rfl⟩
    rfl
  cases lane with
  | center r=>
    change (Reference.scatter c ops a old _).mem "M" (c.carryAddress idx r.val)=_
    rw [←carryEq r]
    unfold Reference.scatter Reference.writeOutput
    rw [State.scatter_frame _ _ _ _ _ _ (Or.inl (by change ("M" : RegionName)≠"O"; decide))]
    unfold Reference.writeRows
    rw [State.scatter_readback _ _ _ _ (r,()) rowInj,finalCenter_encoded c old carry encoded r]
  | denominator r=>
    change c.finish=0 at live
    change (Reference.scatter c ops a old _).mem "L" (c.carryAddress idx r.val)=_
    rw [←carryEq r]
    unfold Reference.scatter Reference.writeOutput
    rw [State.scatter_frame _ _ _ _ _ _ (Or.inl (by change ("L" : RegionName)≠"O"; decide))]
    unfold Reference.writeRows
    rw [State.scatter_frame _ _ _ _ _ _ (Or.inl (by decide)),if_pos live]
    rw [State.scatter_readback _ _ _ _ (r,()) rowInj,encoded.2.1]
    rfl
  | output r d=>
    change (Reference.scatter c ops a old _).mem "O" (c.outAddress idx r.val d.val)=_
    rw [←outEq r d]
    unfold Reference.scatter Reference.writeOutput
    apply Eq.trans (State.scatter_readback (Reference.optr c a).region
      (fun i : TileIndex [c.BM,c.D]=>(Reference.optr c a).address (TileShape.indexToList _ i))
      (fun i=>Cell.mk .real (Reference.finalOutput c ops old i)) _ (r,d,()) outInj)
    exact congrArg (Cell.mk .real) (finalOutput_encoded c old carry encoded r d)

#axiomsClean scatter_result
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
