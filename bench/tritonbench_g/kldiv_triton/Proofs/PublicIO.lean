import bench.tritonbench_g.kldiv_triton.IO
import bench.tritonbench_g.kldiv_triton.Proofs.KernelInputs
import bench.tritonbench_g.kldiv_triton.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
noncomputable section
set_option maxHeartbeats 800000

theorem schedule_run {α : Type} [Inhabited α] (m : Tile.Model α)
    (source : ComputeKernel) (work : List Nat) (initial : State α) :
    runSchedule (work.map (IO.launch m source)) initial =
      GridExecution.sequence (fun pid s => exec m.alg source (GridExecution.atProgram s pid)) work initial := by
  induction work generalizing initial with
  | nil => rfl
  | cons pid rest ih =>
      change (exec m.alg source (GridExecution.atProgram initial pid)).bind
        (runSchedule (rest.map (IO.launch m source))) =
        (exec m.alg source (GridExecution.atProgram initial pid)).bind _
      cases h : exec m.alg source (GridExecution.atProgram initial pid) with
      | none => rfl
      | some final => exact ih final

theorem schedule_safe {α : Type} [Inhabited α] (m : Tile.Model α)
    (source : ComputeKernel) (bounds : RegionName → Nat) (work : List Nat)
    (safe : ∀ pid, pid∈work → ∀ initial,
      TraceSafe m.alg bounds source (GridExecution.atProgram initial pid)) (initial : State α) :
    ScheduleSafe bounds (work.map (IO.launch m source)) initial := by
  induction work generalizing initial with
  | nil => trivial
  | cons pid rest ih =>
      refine ⟨safe pid (by simp) initial,?_⟩
      intro final _
      exact ih (fun p hp => safe p (by simp [hp])) final

theorem forward {α : Type} [Inhabited α] (m : Tile.Model α)
    (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows)
      (cols:=outputCount N B reduction) so))
    (work : List Nat) (order : work.Perm (List.range rows))
    (f : (Nat → α) → (Nat → α) → Nat → α)
    (formula : ∀ ys ts c,KernelInputs.forwardValue m N B flag reduction ys ts c=f ys ts c) :
    (IO.forward m rows sy st so N B flag reduction work).Implements (IO.forwardFormula f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply schedule_safe m _ bounds work _ initial
    intro pid member current
    have hr : pid<rows := List.mem_range.mp (order.mem_iff.mp member)
    apply KernelSafety.forward_source m sy st so N B flag reduction positive bounds
      (GridExecution.atProgram current pid)
    · intro j hj
      exact hb.2.1 (0 : Fin 2) (⟨pid,hr⟩,⟨j,hj⟩,PUnit.unit) trivial
    · intro j hj
      exact hb.2.1 (1 : Fin 2) (⟨pid,hr⟩,⟨j,hj⟩,PUnit.unit) trivial
    · intro j hj
      exact hb.2.2.1 (0 : Fin 1) (⟨pid,hr⟩,⟨j,hj⟩,PUnit.unit) trivial
  · intro vals initial hin
    let ys := IO.rowsOf (vals (0 : Fin 2))
    let ts := IO.rowsOf (vals (1 : Fin 2))
    have hy : ∀ r, r<rows → ∀ c, c<N → Tile.input initial "y_ptr" (r*sy) c=ys r c := by
      intro r hr c hc
      simpa only [ys,IO.rowsOf,hr,hc,dif_pos] using
        hin (0 : Fin 2) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
    have ht : ∀ r, r<rows → ∀ c, c<N → Tile.input initial "gt_ptr" (r*st) c=ts r c := by
      intro r hr c hc
      simpa only [ts,IO.rowsOf,hr,hc,dif_pos] using
        hin (1 : Fin 2) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
    obtain ⟨final,ran,values,frame⟩ := KernelInputs.forward m rows sy st so N B flag reduction
      positive unique ys ts initial hy ht work order
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_run m _ work initial).trans ran
    · intro o j _
      fin_cases o
      have output := values (j.1,j.2.1) trivial
      dsimp only [RawGrid.address] at output
      rw [formula] at output
      change mapCell id (final.mem "loss_ptr" (j.1.val*so+j.2.1.val)) = _
      rw [output]
      rfl
    · intro r offset outside
      apply frame r offset
      by_cases different : r≠"loss_ptr"
      · exact Or.inl different
      · right
        intro x _
        exact (outside.1 (0 : Fin 1) (x.1,x.2,PUnit.unit) trivial).resolve_left different

theorem backward {α : Type} [Inhabited α] (m : Tile.Model α)
    (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows) (cols:=N) so))
    (work : List Nat) (order : work.Perm (List.range rows))
    (f : α → α) (formula : ∀ v,Tile.gradient m flag (m.alg.fp32Load v)=f v) :
    (IO.backward m rows so st N B flag work).Implements (IO.backwardFormula f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply schedule_safe m _ bounds work _ initial
    intro pid member current
    have hr : pid<rows := List.mem_range.mp (order.mem_iff.mp member)
    apply KernelSafety.backward_source m so st N B flag positive bounds
      (GridExecution.atProgram current pid)
    · intro j hj
      exact hb.2.1 (0 : Fin 1) (⟨pid,hr⟩,⟨j,hj⟩,PUnit.unit) trivial
    · intro j hj
      exact hb.2.2.1 (0 : Fin 1) (⟨pid,hr⟩,⟨j,hj⟩,PUnit.unit) trivial
  · intro vals initial hin
    let ts := IO.rowsOf (vals (0 : Fin 1))
    have ht : ∀ r, r<rows → ∀ c, c<N → Tile.input initial "target_ptr" (r*st) c=ts r c := by
      intro r hr c hc
      simpa only [ts,IO.rowsOf,hr,hc,dif_pos] using
        hin (0 : Fin 1) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
    obtain ⟨final,ran,values,frame⟩ := KernelInputs.backward m rows so st N B flag
      positive unique ts initial ht work order
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_run m _ work initial).trans ran
    · intro o j _
      fin_cases o
      have output := values (j.1,j.2.1) trivial
      dsimp only [RawGrid.address] at output
      rw [formula] at output
      change mapCell id (final.mem "input_ptr" (j.1.val*so+j.2.1.val)) = _
      rw [output]
      simp only [mapCell,mapValue,IO.backwardFormula,ts,IO.rowsOf,j.1.isLt,j.2.1.isLt,dif_pos,id_eq]
      rcases j with ⟨r,c,⟨⟩⟩
      rfl
    · intro r offset outside
      apply frame r offset
      by_cases different : r≠"input_ptr"
      · exact Or.inl different
      · right
        intro x _
        exact (outside.1 (0 : Fin 1) (x.1,x.2,PUnit.unit) trivial).resolve_left different

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.PublicIO
