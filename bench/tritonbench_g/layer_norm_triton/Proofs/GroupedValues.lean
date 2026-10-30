import bench.tritonbench_g.layer_norm_triton.Proofs.GroupedReference
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedDX

namespace VeriTile.Bench.Repairs.LayerNormTriton.GroupedValues
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph SourceGroupedExecution SourceGroupedResult
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

structure Inputs (s : State α) (stride M N B : Nat)
    (x dy : Nat → Fin B → α) (w : Fin B → α) (mean inverse : Nat → α) : Prop where
  x : ∀ row,row<M → ∀ j : Fin B,j.val<N → (s.mem "X" (row*stride+j.val)).read .real=x row j
  dy : ∀ row,row<M → ∀ j : Fin B,j.val<N → (s.mem "DY" (row*stride+j.val)).read .real=dy row j
  w : ∀ j : Fin B,j.val<N → (s.mem "W" j.val).read .real=w j
  mean : ∀ row,row<M → (s.mem "Mean" row).read .real=mean row
  inverse : ∀ row,row<M → (s.mem "Rstd" row).read .real=inverse row

theorem after_store (m : Model α) (s : State α) (stride M N B cur : Nat)
    (x dy : Nat → Fin B → α) (w : Fin B → α) (mean inverse : Nat → α)
    (h : Inputs s stride M N B x dy w mean inverse) :
    Inputs (storeDX m s stride N B cur) stride M N B x dy w mean inverse := by
  constructor
  · intro row hr j hj
    rw [storeDX_frame m s stride N B cur "X" _ (.inl (by decide))]
    exact h.x row hr j hj
  · intro row hr j hj
    rw [storeDX_frame m s stride N B cur "DY" _ (.inl (by decide))]
    exact h.dy row hr j hj
  · intro j hj
    rw [storeDX_frame m s stride N B cur "W" _ (.inl (by decide))]
    exact h.w j hj
  · intro row hr
    rw [storeDX_frame m s stride N B cur "Mean" _ (.inl (by decide))]
    exact h.mean row hr
  · intro row hr
    rw [storeDX_frame m s stride N B cur "Rstd" _ (.inl (by decide))]
    exact h.inverse row hr

theorem row_values (m : Model α) (s : State α) (stride M N B row : Nat)
    (x dy : Nat → Fin B → α) (w : Fin B → α) (mean inverse : Nat → α)
    (h : Inputs s stride M N B x dy w mean inverse) (hr : row<M) :
    (∀ j : Fin B, SourceGroupedGraph.dx m s stride N B row j.val=
      GroupedReference.dx m N B (x row) (dy row) w (mean row) (inverse row) j) ∧
    (∀ group a, let b := nextCarries m s stride N B group row a
      let c := GroupedReference.partials m N (x row) (dy row) (mean row) (inverse row)
        (decide (row=group)) (fun j => a.1 (j,()),fun j => a.2 (j,()))
      (∀ j : Fin B, b.1 (j,())=c.1 j) ∧ (∀ j : Fin B,b.2 (j,())=c.2 j)) := by
  have lx (j : Fin B) : loaded m s "X" (row*stride) N j.val=GroupedReference.loaded m N (x row) j := by
    by_cases hj : j.val<N
    · simp [loaded,GroupedReference.loaded,hj,h.x row hr j hj]
    · simp [loaded,GroupedReference.loaded,hj]
  have ld (j : Fin B) : loaded m s "DY" (row*stride) N j.val=GroupedReference.loaded m N (dy row) j := by
    by_cases hj : j.val<N
    · simp [loaded,GroupedReference.loaded,hj,h.dy row hr j hj]
    · simp [loaded,GroupedReference.loaded,hj]
  have xn (j : Fin B) : xhat m s stride N row j.val=GroupedReference.xhat m N (x row) (mean row) (inverse row) j := by
    simp only [xhat,GroupedReference.xhat,lx,SourceGroupedGraph.inverse,h.mean row hr,h.inverse row hr]
  have wd (j : Fin B) : wdy m s stride N row j.val=GroupedReference.wdy m N (dy row) w j := by
    by_cases hj : j.val<N
    · simp [wdy,GroupedReference.wdy,hj,h.w j hj,ld]
    · simp [wdy,GroupedReference.wdy,hj]
  constructor
  · intro j
    simp only [SourceGroupedGraph.dx,GroupedReference.dx,first,second,xn,wd,
      SourceGroupedGraph.inverse,h.inverse row hr]
  · intro group a
    constructor <;> intro j <;>
      simp only [nextCarries,GroupedReference.partials,ld,xn,decide_eq_true_eq]

private theorem fold_read (offset : Fin B → Nat) (values : Fin B → α) (N query : Nat)
    (lanes : List (Fin B)) (s : State α) :
    ((lanes.foldl (fun t j => if j.val<N then t.write "DX" (offset j) (.mk .real (values j)) else t) s).mem "DX" query).read .real=
      lanes.foldl (fun v j => if j.val<N ∧ query=offset j then values j else v) ((s.mem "DX" query).read .real) := by
  induction lanes generalizing s with
  | nil => rfl
  | cons j rest ih =>
    simp only [List.foldl_cons]
    by_cases hj : j.val<N
    · simp only [hj,if_true,true_and]
      rw [ih]
      by_cases hq : query=offset j <;> simp [State.write,hq]
    · simp only [hj,if_false,false_and]
      exact ih s

theorem store_read (m : Model α) (s : State α) (stride M N B row query : Nat)
    (x dy : Nat → Fin B → α) (w : Fin B → α) (mean inverse : Nat → α)
    (h : Inputs s stride M N B x dy w mean inverse) (hr : row<M) :
    ((storeDX m s stride N B row).mem "DX" query).read .real=
      (List.finRange B).foldl (fun v j => if j.val<N ∧ query=row*stride+j.val then
        GroupedReference.dx m N B (x row) (dy row) w (mean row) (inverse row) j else v)
        ((s.mem "DX" query).read .real) := by
  unfold storeDX
  simp only [TileShape.allIndices,List.map_cons,List.map_nil,List.flatMap_cons,
    List.flatMap_nil,List.singleton_append]
  have idx : (List.finRange B).flatMap (fun i => [(i,PUnit.unit)])=
      (List.finRange B).map (fun i => (i,PUnit.unit)) := by
    generalize List.finRange B=l
    induction l with
    | nil => rfl
    | cons i rest ih => simp [ih]
  rw [idx,List.foldl_map]
  rw [fold_read]
  simp only [(row_values m s stride M N B row x dy w mean inverse h hr).1]

end
end VeriTile.Bench.Repairs.LayerNormTriton.GroupedValues
