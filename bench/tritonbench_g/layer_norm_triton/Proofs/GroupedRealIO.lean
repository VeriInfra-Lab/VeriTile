import bench.tritonbench_g.layer_norm_triton.IO
import bench.tritonbench_g.layer_norm_triton.Proofs.GroupedIOResult
import bench.tritonbench_g.layer_norm_triton.Proofs.RealEndpoints
import bench.tritonbench_g.fast_layernorm.Proofs.RealCongruence

namespace VeriTile.Bench.Repairs.LayerNormTriton.GroupedRealIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def extend {B : Nat} (x : Fin B → ℝ) (j : Nat) : ℝ :=
  if hj : j<B then x ⟨j,hj⟩ else 0

def formula (M N G B group : Nat) (vals : Grouped.Inputs ℝ M N B) : Grouped.Outputs ℝ M N B :=
  Grouped.result
    (fun row j => RealProof.Spec.gradient N (extend (Grouped.x vals row.val))
      (extend (Grouped.dy vals row.val)) (extend (Grouped.w vals))
      (Grouped.mean vals row.val) (Grouped.inverse vals row.val) j.val)
    (fun j => ((groupRows M G group).map (fun row => Grouped.dy vals row j *
      ((Grouped.x vals row j-Grouped.mean vals row)*Grouped.inverse vals row))).sum)
    (fun j => ((groupRows M G group).map (fun row => Grouped.dy vals row j)).sum)

private theorem row_lt (M G group row : Nat) (hg : 0<G) (h : row∈groupRows M G group) : row<M := by
  by_cases hc : group<M
  · exact ((Geometry.group_mem M G group row hg hc).mp h).choose_spec.2
  · have hd : (M-group+G-1)/G=0 := by apply Nat.div_eq_of_lt; omega
    simp [groupRows,hd] at h

theorem implements (m : Model ℝ) (laws : SourceForwardReal.Laws m)
    (stride M N G B group : Nat) (groups : 0<G) (covers : N≤B)
    (rowAddresses : ∀ row,row<M → row*stride=row*N) :
    (io m stride M N G B group).Implements (formula M N G B group) := by
  intro vals _ bounds boundsOK initial binding
  have baseBounds : (Grouped.io m stride M N G B group (fun _ _ => id)).InBounds bounds vals :=
    ⟨boundsOK.1,boundsOK.1.2.2.1⟩
  obtain ⟨safe,t,ran,_,frame⟩ := GroupedIOResult.implements m stride M N G B group (fun _ _ => id)
    vals trivial bounds baseBounds initial binding
  let start := (job m (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) group).initial initial
  obtain ⟨u,hu,_,dx,partials,_⟩ := Correctness.layer_norm_grouped_source_real m laws stride M N G B start groups covers rowAddresses
  have hrun : FP.Structural.exec (engine m)
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) start=some t := by
    simpa only [Grouped.io,Grouped.layout,runSchedule_singleton] using ran
  have same : u=t := Option.some.inj (hu.symm.trans hrun)
  subst u
  have hx (row : Nat) (hr : row<M) (j : Nat) (hj : j<N) :
      SourceGroupedReal.row start "X" stride row j=extend (Grouped.x vals row) j := by
    simpa [SourceGroupedReal.row,Grouped.x,extend,show j<B by omega,hr,
      Grouped.io,Grouped.layout,Grouped.inputs,mapValue,start,Invocation.initial,bindRegisters]
      using binding (0 : Fin 6) (⟨row,hr⟩,⟨j,by omega⟩,()) hj
  have hy (row : Nat) (hr : row<M) (j : Nat) (hj : j<N) :
      SourceGroupedReal.row start "DY" stride row j=extend (Grouped.dy vals row) j := by
    simpa [SourceGroupedReal.row,Grouped.dy,extend,show j<B by omega,hr,
      Grouped.io,Grouped.layout,Grouped.inputs,mapValue,start,Invocation.initial,bindRegisters]
      using binding (1 : Fin 6) (⟨row,hr⟩,⟨j,by omega⟩,()) hj
  have hw (j : Nat) (hj : j<N) : SourceGroupedReal.weight start j=extend (Grouped.w vals) j := by
    simpa [SourceGroupedReal.weight,Grouped.w,extend,show j<B by omega,
      Grouped.io,Grouped.layout,Grouped.inputs,mapValue,start,Invocation.initial,bindRegisters]
      using binding (2 : Fin 6) (⟨j,by omega⟩,()) hj
  have hm (row : Nat) (hr : row<M) : SourceGroupedReal.savedMean start row=Grouped.mean vals row := by
    simpa [SourceGroupedReal.savedMean,Grouped.mean,hr,
      Grouped.io,Grouped.layout,Grouped.inputs,mapValue,start,Invocation.initial,bindRegisters]
      using binding (3 : Fin 6) (⟨row,hr⟩,()) trivial
  have hi (row : Nat) (hr : row<M) : SourceGroupedReal.savedInverse start row=Grouped.inverse vals row := by
    simpa [SourceGroupedReal.savedInverse,Grouped.inverse,hr,
      Grouped.io,Grouped.layout,Grouped.inputs,mapValue,start,Invocation.initial,bindRegisters]
      using binding (4 : Fin 6) (⟨row,hr⟩,()) trivial
  refine ⟨safe,t,ran,?_,frame⟩
  intro port lane active
  fin_cases port
  · have v := dx lane.1.val active lane.2.1.val lane.2.1.isLt
    have addr := rowAddresses lane.1.val lane.1.isLt
    have xr : SourceGroupedReal.row start "X" N lane.1.val=SourceGroupedReal.row start "X" stride lane.1.val := by
      funext j
      simp only [SourceGroupedReal.row,addr]
    have yr : SourceGroupedReal.row start "DY" N lane.1.val=SourceGroupedReal.row start "DY" stride lane.1.val := by
      funext j
      simp only [SourceGroupedReal.row,addr]
    change (t.mem "DX" (lane.1.val*stride+lane.2.1.val)).read .real=_
    rw [addr,v,Cell.read_same,xr,yr,hm _ lane.1.isLt,hi _ lane.1.isLt]
    exact FastLayernorm.RealProof.gradient_congr N _ _ _ _ _ _ _ _ _ lane.2.1.isLt
      (hx _ lane.1.isLt) (hy _ lane.1.isLt) hw
  · have v := (partials lane active).1
    change t.mem "DW" (group*N+lane.1.val)=_ at v
    change (t.mem "DW" (group*N+lane.1.val)).read .real=_
    rw [v,Cell.read_same]
    change ((groupRows M G group).map _).sum=((groupRows M G group).map _).sum
    congr 1
    apply List.map_congr_left
    intro row member
    have hr := row_lt M G group row groups member
    simp only [SourceGroupedSums.weightTerm,SourceGroupedReal.normalized,hy row hr lane.1.val active,
      hx row hr lane.1.val active,hm row hr,hi row hr,extend,dif_pos lane.1.isLt]
  · have v := (partials lane active).2
    change t.mem "DB" (group*N+lane.1.val)=_ at v
    change (t.mem "DB" (group*N+lane.1.val)).read .real=_
    rw [v,Cell.read_same]
    change ((groupRows M G group).map _).sum=((groupRows M G group).map _).sum
    congr 1
    apply List.map_congr_left
    intro row member
    have hr := row_lt M G group row groups member
    simp only [SourceGroupedSums.biasTerm,hy row hr lane.1.val active,extend,dif_pos lane.1.isLt]
#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormTriton.GroupedRealIO
