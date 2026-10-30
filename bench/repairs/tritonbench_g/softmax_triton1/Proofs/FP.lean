import bench.repairs.tritonbench_g.softmax_triton1.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Floating
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Physical FP32 loads preserve the already decoded scalar value. -/
def backend {α} (M : Algebra α) : Algebra α := {M with fp32Load := id}

/-- Explicit FP32 operations from the shared source. Max and sum stay opaque
on the complete padded tile in source order; tl.exp is retained. The structural
`.real` channel is an opaque scalar carrier, not mathematical Real arithmetic. -/
def rowExpression {α} (M : Algebra α) (n B : Nat) (xs : Fin B → α) (i : Fin B) : α :=
  let row : Values α .real [B] := fun j => if j.1.val<n then xs j.1 else M.negInf
  let maximum := M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false row PUnit.unit
  let numerator : Values α .real [B] := fun j =>
    M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub (row j) maximum)
  let denominator := M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false numerator PUnit.unit
  M.binary (some .fp32) .real .div (numerator (i,PUnit.unit)) denominator

theorem expression_congr {α} (M : Algebra α) (n B : Nat) (xs ys : Fin B → α)
    (same : ∀ j, j.val<n → xs j=ys j) (i : Fin B) :
    rowExpression M n B xs i=rowExpression M n B ys i := by
  have eq : (fun j : TileIndex [B] => if j.1.val<n then xs j.1 else M.negInf)=
      (fun j : TileIndex [B] => if j.1.val<n then ys j.1 else M.negInf) := by
    funext j
    by_cases h : j.1.val<n <;> simp [h,same j.1]
  unfold rowExpression
  rw [eq]

theorem masked_readback_active {α} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : State α)
    (i : TileIndex shape) (hi : active i)
    (unique : ∀ j, active j → off j=off i → j=i) :
    ((TileShape.allIndices shape).foldl
      (fun t j => if active j then t.write r (off j) (val j) else t) s).mem r (off i)=val i := by
  rw [State.masked_scatter_eq_filter]
  let l := (TileShape.allIndices shape).filter (fun j => decide (active j))
  have hmem : i∈l := by simp [l,TileShape.mem_allIndices,hi]
  have hnd : l.Nodup := (TileShape.allIndices_nodup shape).filter _
  obtain ⟨before,after,hl⟩ := List.append_of_mem hmem
  rw [hl,List.nodup_append,List.nodup_cons] at hnd
  obtain ⟨_,⟨hnot,_⟩,_⟩ := hnd
  change (l.foldl _ s).mem r (off i)=_
  rw [hl,List.foldl_append,List.foldl_cons,State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj heq
    have ha : active j := by
      have hjl : j∈l := by rw [hl];simp [hj]
      simpa [l] using hjl
    exact hnot ((unique j ha heq.symm) ▸ hj)

theorem row_exec {α} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0<B) (s : State α) :
    ∃ t, exec (backend M) (softmax_kernel out inp ib ob ir ic orow oc n B) s=some t := by
  simp [exec,softmax_kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalOp,store,
    State.setReg,numeric,numericLt,natLt,bop,TileShape.axisDim,hB,Option.bind,Function.comp_def,
    evalComputeOp,backend,ComputeDType.eraseDType,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]

theorem row_value {α} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0<B)
    (inj : ∀ i j : Fin B, i.val<n → j.val<n → i.val*oc=j.val*oc → i=j)
    (s t : State α) (runEq : exec (backend M) (softmax_kernel out inp ib ob ir ic orow oc n B) s=some t)
    (i : Fin B) (hi : i.val<n) :
    t.mem out (ob+s.pids 0*orow+i.val*oc)=.mk .real
      (rowExpression M n B (fun j => (s.mem inp (ib+s.pids 0*ir+j.val*ic)).read .real) i) := by
  simp [exec,softmax_kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalOp,store,
    State.setReg,numeric,numericLt,natLt,bop,TileShape.axisDim,hB,Option.bind,Function.comp_def,
    evalComputeOp,backend,ComputeDType.eraseDType,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at runEq
  subst t
  rw [masked_readback_active (shape := [B]) out
    (fun j => ob+s.pids 0*orow+j.1.val*oc) _ (fun j => j.1.val<n) _
    (i,PUnit.unit) hi (by
      intro j hj eq
      have same : j.1=i := inj j.1 i hj hi (by simpa using eq)
      cases j with | mk k u => cases u;simp_all)]
  rfl

theorem row_frame {α} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0<B)
    (s t : State α) (runEq : exec (backend M) (softmax_kernel out inp ib ob ir ic orow oc n B) s=some t)
    (r : RegionName) (a : Nat)
    (miss : ∀ i : Fin B, i.val<n → ¬(out=r ∧ ob+s.pids 0*orow+i.val*oc=a)) :
    t.mem r a=s.mem r a := by
  simp [exec,softmax_kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalOp,store,
    State.setReg,numeric,numericLt,natLt,bop,TileShape.axisDim,hB,Option.bind,Function.comp_def,
    evalComputeOp,backend,ComputeDType.eraseDType,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at runEq
  subst t
  apply State.masked_scatter_frame
  by_cases eq : r=out
  · right
    intro i _ hi off
    exact miss i.1 hi ⟨eq.symm,off.symm⟩
  · exact Or.inl eq

#axiomsClean row_exec
#axiomsClean row_value
#axiomsClean row_frame
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Floating
