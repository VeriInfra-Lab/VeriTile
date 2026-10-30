import bench.tritonbench_g.bgmv_shrink_kernel.IO

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOGeometry
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
set_option maxHeartbeats 1000000
local instance (h : Call) : DecidableEq (Program h) := Classical.decEq _

theorem programs_nodup (h : Call) : (IO.programs h).Nodup := by
  rw [IO.programs,List.nodup_flatMap]
  constructor
  · intro b _
    exact (List.nodup_finRange 64).map (fun _ _ same => congrArg Program.split same)
  · apply (List.nodup_finRange h.batches).imp
    intro a b different p left right
    obtain ⟨i,_,rfl⟩ := List.mem_map.mp left
    obtain ⟨j,_,same⟩ := List.mem_map.mp right
    exact different (congrArg Program.batch same).symm

theorem row_mem (h : Call) (b : Fin h.batches) (p : Program h) :
    p ∈ (List.finRange 64).map (fun split => (⟨b,split⟩ : Program h)) ↔ p.batch = b := by
  rcases p with ⟨row,split⟩
  simp [Program.mk.injEq,eq_comm]

theorem row_partition (h : Call) (b : Fin h.batches) :
    let row := (List.finRange 64).map (fun split => (⟨b,split⟩ : Program h))
    (row ++ (IO.programs h).filter (fun p => decide (p ∉ row))).Perm (IO.programs h) := by
  classical
  dsimp only
  let row := (List.finRange 64).map (fun split => (⟨b,split⟩ : Program h))
  have nodup : row.Nodup :=
    (List.nodup_finRange 64).map (fun _ _ same => congrArg Program.split same)
  have same : row.Perm ((IO.programs h).filter (fun p => decide (p ∈ row))) := by
    apply (List.perm_ext_iff_of_nodup nodup ((programs_nodup h).filter _)).mpr
    intro p
    simp
  have partition := List.filter_append_perm (fun p : Program h => decide (p ∈ row)) (IO.programs h)
  simpa only [decide_not] using (same.append_right _).trans partition

theorem output_unique (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h) (observe : α → γ)
    (vals : (IO.programIO M P h indices scale orders observe).Inputs) :
    (IO.programIO M P h indices scale orders observe).toMemoryIO.OutputUnique vals := by
  intro left right _ _ same
  rcases left with ⟨o,left⟩
  rcases right with ⟨p,right⟩
  have ho : o = 0 := Fin.eq_zero o
  have hp : p = 0 := Fin.eq_zero p
  subst o
  subst p
  obtain ⟨b,n,unit⟩ := left
  obtain ⟨d,m,unit'⟩ := right
  have address : outputOffset h b.val n.val = outputOffset h d.val m.val := congrArg Prod.snd same
  obtain ⟨row,column⟩ := output_injective h b.val d.val n m address
  have rowEq : b=d := Fin.ext row
  subst d
  subst m
  cases unit
  cases unit'
  rfl

theorem schedule_complete (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h) (observe : α → γ)
    (vals : (IO.programIO M P h indices scale orders observe).Inputs) :
    (IO.programIO M P h indices scale orders observe).ScheduleComplete vals := by
  classical
  refine ⟨rfl,output_unique M P h indices scale orders observe vals,?_⟩
  intro key _
  rcases key with ⟨o,j⟩
  have ho : o = 0 := Fin.eq_zero o
  subst o
  obtain ⟨b,n,unit⟩ := j
  change (if indices b.val = -1 then [] else (orders.order b n).map (fun split => (⟨b,split⟩ : Program h))).Perm
      (if indices b.val = -1 then [] else (List.finRange 64).map (fun split => (⟨b,split⟩ : Program h))) ∧
    ((if indices b.val = -1 then [] else (List.finRange 64).map (fun split => (⟨b,split⟩ : Program h))) ++
      (IO.programs h).filter (fun p => decide (p ∉ (if indices b.val = -1 then [] else
        (List.finRange 64).map (fun split => (⟨b,split⟩ : Program h)))))).Perm (IO.programs h)
  by_cases sentinel : indices b.val = -1
  · simp [sentinel]
  · simp only [sentinel,if_false]
    exact ⟨(orders.complete b n).map _,row_partition h b⟩

#axiomsClean schedule_complete
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOGeometry
