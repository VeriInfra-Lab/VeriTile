import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineFrame
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeScheduleReturn

/-! Disjoint physical row execution. The merge reads actual final row cells;
each row's complete physical frame and pairwise disjointness are proved. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI AliasMemory
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

def WritesAt {α : Type} (c : Config α block) (row : Fin c.M) (address : Nat) : Prop :=
  ∃ key : Key, Contract.written c.flags c.regions c.strides c.M c.N row.val key.1 key.2 ∧
    (layout c.allocations).address key=address

def executeGrid {α : Type} [Inhabited α] (c : Config α block) (physical : Memory c) : Memory c :=
  fun i => if h : ∃ row : Fin c.M, WritesAt c row i.val then execute c h.choose physical i else physical i

theorem physical_row_unique {α : Type} (c : Config α block)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N)
    (i j : Fin c.M) (address : Nat) (hi : WritesAt c i address) (hj : WritesAt c j address) : i=j := by
  obtain ⟨ki,hki,hai⟩ := hi
  obtain ⟨kj,hkj,haj⟩ := hj
  have hwi : Writes c.flags c.regions c.strides c.M c.N ki := ⟨i.val,i.isLt,hki⟩
  have hwj : Writes c.flags c.regions c.strides c.M c.N kj := ⟨j.val,j.isLt,hkj⟩
  have he : kj=ki := (c.storage.fresh ki hwi).2 kj (c.storage.fresh kj hwj).1 (haj.trans hai.symm)
  subst kj
  exact Fin.ext (NativeGeometry.written_row_unique c.flags c.regions c.strides c.M c.N i.val j.val hgeo
    i.isLt j.isLt ki.1 ki.2 hki hkj)

theorem grid_readback {α : Type} [Inhabited α] (c : Config α block)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N) (physical : Memory c) (row : Fin c.M)
    (i : Fin (layout c.allocations).capacity) (hi : WritesAt c row i.val) :
    executeGrid c physical i=execute c row physical i := by
  have hex : ∃ row : Fin c.M, WritesAt c row i.val := ⟨row,hi⟩
  rw [executeGrid,dif_pos hex,physical_row_unique c hgeo hex.choose row i.val hex.choose_spec hi]

theorem grid_frame {α : Type} [Inhabited α] (c : Config α block) (physical : Memory c)
    (i : Fin (layout c.allocations).capacity) (hi : ∀ row : Fin c.M, Outside c row i) :
    executeGrid c physical i=physical i := by
  apply dif_neg
  rintro ⟨row,key,hk,ha⟩
  exact hi row key.1 key.2 hk ha

/-- Whole physical grid simulation to the checked native schedule. The
launch words/control/access bridge is inherited from each executed row. -/
theorem grid_correspondence {α : Type} [Inhabited α] (c : Config α block)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N) (physical : Memory c) (template : State α) :
    view c (executeGrid c physical) template=
      NativeSchedule.executeGrid c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides c.M c.N
        c.eps c.p c.accepted.blockPositive (view c physical template) := by
  apply TypedPhysical.state_ext
  · funext region offset
    let key : Key := (region,offset)
    change TypedPhysical.read (layout c.allocations) (executeGrid c physical) key=
      if h : ∃ row : Fin c.M, Contract.written c.flags c.regions c.strides c.M c.N row.val region offset then
        (NativeSchedule.execute c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides c.M c.N h.choose.val
          c.eps c.p c.accepted.blockPositive (view c physical template)).mem region offset
      else TypedPhysical.read (layout c.allocations) physical key
    by_cases hw : ∃ row : Fin c.M, Contract.written c.flags c.regions c.strides c.M c.N row.val region offset
    · rw [dif_pos hw]
      have hlive := (c.storage.fresh key ⟨hw.choose.val,hw.choose.isLt,hw.choose_spec⟩).1
      let ix : Fin (layout c.allocations).capacity := ⟨(layout c.allocations).address key,(layout c.allocations).bounded key hlive⟩
      have hread := grid_readback c hgeo physical hw.choose ix ⟨key,hw.choose_spec,rfl⟩
      have hrow := congrArg (fun s : State α => s.mem region offset) (execute_correspondence c hw.choose physical template)
      change TypedPhysical.read (layout c.allocations) (execute c hw.choose physical) key=_ at hrow
      rw [TypedPhysical.read,dif_pos hlive] at hrow ⊢
      exact hread.trans hrow
    · rw [dif_neg hw]
      by_cases hlive : (layout c.allocations).live key
      · let ix : Fin (layout c.allocations).capacity := ⟨(layout c.allocations).address key,(layout c.allocations).bounded key hlive⟩
        have hout : ∀ row : Fin c.M, Outside c row ix := by
          intro row r o hwrite ha
          have hkey : key=(r,o) := (c.storage.fresh (r,o) ⟨row.val,row.isLt,hwrite⟩).2 key hlive ha.symm
          have he : r=region ∧ o=offset := by simpa only [key,Prod.mk.injEq] using hkey.symm
          exact hw ⟨row,by simpa only [he.1,he.2] using hwrite⟩
        simpa only [TypedPhysical.read,dif_pos hlive] using grid_frame c physical ix hout
      · simp only [TypedPhysical.read,dif_neg hlive]
  · rfl
  · rfl
  · rfl
  · rfl

theorem empty_grid {α : Type} [Inhabited α] (c : Config α block) (hM : c.M=0) (physical : Memory c) :
    executeGrid c physical=physical := by
  funext i
  apply grid_frame
  intro row
  exact False.elim (by have h := row.isLt; omega)

/-- Independent eight-field FP readback of the actual word-addressed native
grid, together with all untouched physical cells. -/
theorem grid_return {α : Type} [Inhabited α] (c : Config α block)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N) (physical : Memory c) (template : State α) :
    TypedSource.observe c.flags c.regions c.strides c.M c.N (view c (executeGrid c physical) template)=
      NativeSchedule.oracle c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides c.M c.N
        c.eps c.p (view c physical template) ∧
    (∀ i : Fin (layout c.allocations).capacity, (∀ row : Fin c.M, Outside c row i) →
      executeGrid c physical i=physical i) := by
  refine ⟨?_,fun i hi => grid_frame c physical i hi⟩
  rw [grid_correspondence c hgeo physical template]
  have hp := (NativeSchedule.execute_grid_post c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides
    c.M c.N c.eps c.p c.accepted.blockPositive (view c physical template) hsep hgeo).2
  exact NativeSchedule.observe_eq_oracle c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides
    c.M c.N c.eps c.p c.accepted.width _ _ hsep hp

#axiomsClean physical_row_unique
#axiomsClean grid_readback
#axiomsClean grid_frame
#axiomsClean grid_correspondence
#axiomsClean empty_grid
#axiomsClean grid_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
