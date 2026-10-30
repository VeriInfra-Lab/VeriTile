import bench.repairs.tritonbench_g.matmul_triton1.Proofs.Bytes

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton FP.Structural
open Revision001 SharedContract TensorBinding
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem source_store_inside {α : Type} (x : Launch) (s : State α)
    (i : Core.CIndex x.parameters) (hi : Core.active x.parameters s i) :
    Core.cOffset x.parameters s i < x.c.metadata.numel := by
  apply Core.active_c_bound x.parameters s x.c.metadata.numel
  · refine ⟨?_,?_⟩
    · exact (output_numel x).ge
    · intro r c _ _
      simp [Launch.parameters]
  · exact hi

theorem all_output_stored {α : Type} (x : Launch) (s : State α) (n : Nat)
    (hn : n < x.c.metadata.numel) :
    ∃ pid, pid < Geometry.count (Core.geometry x.parameters) ∧
      ∃ i : Core.CIndex x.parameters,
        Core.active x.parameters (GridExecution.atProgram s pid) i ∧
        Core.cOffset x.parameters (GridExecution.atProgram s pid) i = n := by
  obtain ⟨cell,hcell⟩ := index_covers x n hn
  obtain ⟨pid,hpid,i,hr,hc⟩ := Geometry.covers (Core.geometry x.parameters) (geometry x) cell
  have active := Geometry.owns_active (Core.geometry x.parameters) pid cell i hr hc
  refine ⟨pid,hpid,i,(Core.active_at x.parameters s pid hpid i).mpr active,?_⟩
  rw [Core.offset_at x.parameters s pid hpid i]
  change Geometry.outRow (Core.geometry x.parameters) pid i * _ +
    Geometry.outColumn (Core.geometry x.parameters) pid i * _ = _
  rw [hr,hc]
  simpa [Launch.parameters,Core.geometry,offset] using hcell

theorem physical_writers_unique (x : Launch) (p q : Nat)
    (hp : p < Geometry.count (Core.geometry x.parameters))
    (hq : q < Geometry.count (Core.geometry x.parameters))
    (i j : Core.CIndex x.parameters)
    (hi : Geometry.Active (Core.geometry x.parameters) p i)
    (hj : Geometry.Active (Core.geometry x.parameters) q j)
    (b e : Fin 4)
    (equal : address x.c (Geometry.offset (Core.geometry x.parameters) p i)+b.val =
      address x.c (Geometry.offset (Core.geometry x.parameters) q j)+e.val) :
    p = q ∧ i = j ∧ b = e := by
  have hb := b.isLt
  have he := e.isLt
  have indices : Geometry.offset (Core.geometry x.parameters) p i =
      Geometry.offset (Core.geometry x.parameters) q j := by unfold address at equal; omega
  have bytes : b = e := Fin.ext (by unfold address at equal; omega)
  let ci := Geometry.index (Core.geometry x.parameters) p i hi
  let cj := Geometry.index (Core.geometry x.parameters) q j hj
  have same : ci = cj := (geometry x).writable indices
  have ownedP : Geometry.Owns (Core.geometry x.parameters) p ci := ⟨i,rfl,rfl⟩
  have ownedQ : Geometry.Owns (Core.geometry x.parameters) q ci := by
    rw [same]
    exact ⟨j,rfl,rfl⟩
  have programs := Geometry.owner_unique (Core.geometry x.parameters) (geometry x) p q hp hq ci ownedP ownedQ
  subst q
  exact ⟨rfl,Geometry.offset_injective (Core.geometry x.parameters) (geometry x) p i j hi hj indices,bytes⟩

theorem actual_output_pointer (x : Launch) (i : Nat) (hi : i < x.c.metadata.numel) :
    (x.c.dataPtr + FiniteIndex.word (i*4)).toNat = address x.c i ∧
    (x.c.dataPtr + FiniteIndex.word (i*4)).toNat % 4 = 0 ∧
    (x.c.dataPtr + FiniteIndex.word (i*4)).toNat+3 < base x.c+4*x.c.metadata.numel := by
  have h := TensorBinding.pointer_window x.c x.lc x.outputContiguous x.outputAligned i hi
  have baseEq : (VeriTile.Torch.FP32.dataView x.c x.lc x.outputContiguous).base = base x.c := by
    simp [VeriTile.Torch.FP32.DataView.base,VeriTile.Torch.FP32.dataView,base,
      show x.c.metadata.numel ≠ 0 by omega]
  simpa only [data_address x.c x.lc x.outputContiguous i hi,baseEq,Nat.mul_comm] using h

/-- Every source store in a program commutes with an ordinary physical word
update, including its original lane order. The initial image is proved equal
to the caller reader by `image_initial`. -/
theorem fold_projection {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (initial : State α) (P : Core.Primitives α) (events : List (Core.CIndex x.parameters))
    (s : State α) :
    image x read ((events.foldl (fun t i => if Core.active x.parameters initial i then
      t.write x.rc (Core.cOffset x.parameters initial i)
        (.mk .real (Core.accumulation P x.parameters initial (Core.steps x.parameters) i)) else t) s).mem) =
    events.foldl (fun mem i => if Core.active x.parameters initial i then
      Function.update mem (address x.c (Core.cOffset x.parameters initial i))
        (Core.accumulation P x.parameters initial (Core.steps x.parameters) i) else mem)
      (image x read s.mem) := by
  induction events generalizing s with
  | nil => rfl
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases active : Core.active x.parameters initial i
    · simp only [if_pos active]
      rw [ih,image_write x read s _ (source_store_inside x initial i active)]
    · simp only [if_neg active]
      exact ih s

theorem program_physical {α : Type} [Inhabited α] (rest : Algebra α)
    (P : Core.Primitives α) (x : Launch) (read : Memory α) (s : State α) :
    ∃ final, FP.Structural.exec (Core.backend rest P) (source x.parameters) s = some final ∧
      image x read final.mem =
      (TileShape.allIndices [x.parameters.blockRows,x.parameters.blockColumns]).foldl
        (fun mem i => if Core.active x.parameters s i then
          Function.update mem (address x.c (Core.cOffset x.parameters s i))
            (Core.accumulation P x.parameters s (Core.steps x.parameters) i) else mem)
        (image x read s.mem) := by
  obtain ⟨final,exec,mem⟩ := FiniteSource.source_memory rest P x.parameters s
    (by norm_num [Launch.parameters])
  refine ⟨final,exec,?_⟩
  rw [mem]
  exact fold_projection x read s P _ s

#axiomsClean source_store_inside
#axiomsClean all_output_stored
#axiomsClean physical_writers_unique
#axiomsClean actual_output_pointer
#axiomsClean program_physical
end
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
