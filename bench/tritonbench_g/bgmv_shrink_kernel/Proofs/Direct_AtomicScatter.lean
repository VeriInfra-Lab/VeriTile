import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_FPValues

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter
open VeriTile.Triton FP.Structural
noncomputable section
variable {α ι : Type}

/-- Within one program, different n lanes have different output addresses.
Thus each RMW reads the entry cell at that address. This structural lemma
assumes no identity whatsoever about the RMW's local numerical function. -/
theorem distinct_fold (r : RegionName) (off : ι → Nat) (modify : ι → Cell α → Cell α)
    (xs : List ι) (unique : xs.Nodup) (injective : Function.Injective off)
    (initial s : State α) (agree : ∀ i∈xs,s.mem r (off i)=initial.mem r (off i)) :
    xs.foldl (fun t i => t.write r (off i) (modify i (t.mem r (off i)))) s=
      xs.foldl (fun t i => t.write r (off i) (modify i (initial.mem r (off i)))) s := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    obtain ⟨absent,uniqueRest⟩ := List.nodup_cons.mp unique
    rw [List.foldl_cons,List.foldl_cons,agree i (by simp)]
    apply ih uniqueRest
    intro j member
    rw [State.write_other]
    · exact agree j (by simp [member])
    · right
      intro same
      exact absent ((injective same) ▸ member)

/-- Exact masked RMW fold to ordinary scatter, retaining the original cells
inside the local primitive. The source grid may still collide across splits. -/
theorem masked (r : RegionName) (off : ι → Nat) (modify : ι → Cell α → Cell α)
    (active : ι → Bool) (xs : List ι) (unique : xs.Nodup) (injective : Function.Injective off)
    (s : State α) :
    xs.foldl (fun t i => if active i then t.write r (off i) (modify i (t.mem r (off i))) else t) s=
      (xs.filter active).foldl (fun t i => t.write r (off i) (modify i (s.mem r (off i)))) s := by
  rw [←List.foldl_filter]
  exact distinct_fold r off modify (xs.filter active) (unique.filter active) injective s s (by simp)

/-- Atomic state folds depend on memory alone; register setup is irrelevant
to readback and the memory frame after the operand tile has been evaluated. -/
theorem memory_congr (r : RegionName) (off : ι → Nat) (modify : ι → Cell α → Cell α)
    (active : ι → Bool) (xs : List ι) (s t : State α) (same : s.mem=t.mem) :
    (xs.foldl (fun u i => if active i then u.write r (off i) (modify i (u.mem r (off i))) else u) s).mem=
    (xs.foldl (fun u i => if active i then u.write r (off i) (modify i (u.mem r (off i))) else u) t).mem := by
  induction xs generalizing s t with
  | nil => exact same
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases yes : active i=Bool.true
    · simp only [yes,if_true]
      apply ih
      simp only [State.write,same]
    · simp only [yes,if_false]
      exact ih s t same

theorem fold_memory (F G : State α → ι → State α) (xs : List ι) (s t : State α)
    (same : s.mem=t.mem)
    (step : ∀ i u v,u.mem=v.mem → (F u i).mem=(G v i).mem) :
    (xs.foldl F s).mem=(xs.foldl G t).mem := by
  induction xs generalizing s t with
  | nil => exact same
  | cons i rest ih => exact ih (F s i) (G t i) (step i s t same)

theorem memory_congr_prop (r : RegionName) (off : ι → Nat) (modify : ι → Cell α → Cell α)
    (active : ι → Prop) [DecidablePred active] (xs : List ι) (s t : State α) (same : s.mem=t.mem) :
    (xs.foldl (fun u i => if active i then u.write r (off i) (modify i (u.mem r (off i))) else u) s).mem=
    (xs.foldl (fun u i => if active i then u.write r (off i) (modify i (u.mem r (off i))) else u) t).mem := by
  simpa only [decide_eq_true_eq] using memory_congr r off modify (fun i => decide (active i)) xs s t same

def tile (h : Call) (P : FPValues.Primitives α) (batch : Nat) (values : Fin h.BN → α)
    [Inhabited α] (s : State α) : State α :=
  (TileShape.allIndices [h.BN]).foldl (fun t i => if i.1.val<h.N then
    t.write "Y" (Geometry.outputOffset h batch i.1.val)
      (.mk h.outputType.toTileDType (ofFloat _ (P.atomic h.outputType
        (toFloat _ ((t.mem "Y" (Geometry.outputOffset h batch i.1.val)).read h.outputType.toTileDType)) (values i.1))))
    else t) s

theorem offset_injective (h : Call) (batch : Nat) :
    Function.Injective (fun i : TileIndex [h.BN] => Geometry.outputOffset h batch i.1.val) := by
  intro i j same
  apply Prod.ext
  · apply Fin.ext
    simpa only [Geometry.outputOffset,add_right_inj] using same
  · exact Subsingleton.elim _ _

theorem tile_scatter (h : Call) (P : FPValues.Primitives α) (batch : Nat) (values : Fin h.BN → α)
    [Inhabited α] (s : State α) :
    tile h P batch values s=
    ((TileShape.allIndices [h.BN]).filter (fun i => decide (i.1.val<h.N))).foldl
      (fun t i => t.write "Y" (Geometry.outputOffset h batch i.1.val)
        (.mk h.outputType.toTileDType (ofFloat _ (P.atomic h.outputType
          (toFloat _ ((s.mem "Y" (Geometry.outputOffset h batch i.1.val)).read h.outputType.toTileDType))
          (values i.1))))) s := by
  simpa only [tile,decide_eq_true_eq] using masked "Y"
    (fun i : TileIndex [h.BN] => Geometry.outputOffset h batch i.1.val)
    (fun i old => .mk h.outputType.toTileDType (ofFloat _ (P.atomic h.outputType
      (toFloat _ (old.read h.outputType.toTileDType)) (values i.1))))
    (fun i => decide (i.1.val<h.N)) _ (TileShape.allIndices_nodup _) (offset_injective h batch) s

theorem tile_value (h : Call) (P : FPValues.Primitives α) (batch : Nat) (values : Fin h.BN → α)
    [Inhabited α] (s : State α) (n : Fin h.N) :
    (tile h P batch values s).mem "Y" (Geometry.outputOffset h batch n.val)=
      .mk h.outputType.toTileDType (ofFloat _ (P.atomic h.outputType
        (toFloat _ ((s.mem "Y" (Geometry.outputOffset h batch n.val)).read h.outputType.toTileDType))
        (values ⟨n.val,n.isLt.trans_le h.rowsCovered⟩))) := by
  rw [tile_scatter]
  refine State.scatter_readback_list "Y" _ _ _ s ((⟨n.val,n.isLt.trans_le h.rowsCovered⟩ : Fin h.BN),PUnit.unit) ?_ ?_ ?_
  · exact (TileShape.allIndices_nodup _).filter _
  · exact List.mem_filter.mpr ⟨TileShape.mem_allIndices [h.BN] _,by simpa using n.isLt⟩
  · exact offset_injective h batch

theorem tile_frame (h : Call) (P : FPValues.Primitives α) (batch : Nat) (values : Fin h.BN → α)
    [Inhabited α] (s : State α) (r : RegionName) (off : Nat)
    (outside : ¬(r="Y" ∧ ∃ n : Fin h.N,off=Geometry.outputOffset h batch n.val)) :
    (tile h P batch values s).mem r off=s.mem r off := by
  rw [tile_scatter]
  apply State.scatter_frame
  by_cases region : r="Y"
  · right
    intro i member same
    apply outside
    exact ⟨region,⟨i.1.val,by simpa only [List.mem_filter,decide_eq_true_eq] using (List.mem_filter.mp member).2⟩,same⟩
  · exact Or.inl region

#axiomsClean tile_value
#axiomsClean tile_frame
#axiomsClean memory_congr
#axiomsClean distinct_fold
#axiomsClean masked
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter
