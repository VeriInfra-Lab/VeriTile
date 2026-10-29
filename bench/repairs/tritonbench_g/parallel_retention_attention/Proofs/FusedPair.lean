import bench.repairs.tritonbench_g.parallel_retention_attention.PairIO
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.FusedKernel

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
open VeriTile Triton FP.Structural Kernel Kernel.Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (withDot)
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type}






theorem forward_inv_run (s : Shape) (tile : Tile) (state : State (Expr α)) :
    (PublicIO.forwardInvocation α s tile).run state=(forwardInvocation s tile).run state :=
  forward_run algebra dots (config s 128) _

theorem backward_inv_run (s : Shape) (tile : Tile) (state : State (Expr α)) :
    (PublicIO.backwardInvocation α s tile).run state=(backwardInvocation s tile).run state :=
  shell_run algebra dots (decodedBackwardConfig s (PublicIO.programIds s tile)) "dq" _

theorem forward_result (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) (xs : ChannelValues α (PublicIO.forwardInputs s))
    (state : State (Expr α)) :
    ∃ final,runSchedule [PublicIO.forwardInvocation α s tile] state=some final ∧
      (PublicIO.forwardIO a p s tile).Frame xs state final := by
  obtain ⟨final,ran,memory⟩:=PublicIO.forward_run s tile valid state
  refine ⟨final,ran,?_⟩
  intro R off outside
  rw [memory]
  apply PublicIO.storeTile_frame_full
  intro i active
  exact outside.1 (0 : Fin 1) i active

theorem backward_result (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (xs : ChannelValues α (PublicIO.backwardInputs s))
    (state : State (Expr α)) :
    ∃ final,runSchedule [PublicIO.backwardInvocation α s tile] state=some final ∧
      (PublicIO.backwardIO a p s tile).Frame xs state final := by
  obtain ⟨final,ran,memory⟩:=PublicIO.backward_run s tile valid state
  let start := (PublicIO.backwardInvocation α s tile).initial state
  let q := FPQuery.values algebra dots (backwardConfig s tile) start
  let kv := FPStages.backwardValues algebra dots (backwardConfig s tile) start
  let qmem := storeTile "dq" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
    (tile.chunk*64) (tile.key*s.BK) start q
  let kmem := storeTile "dk" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
    (tile.chunk*64) (tile.key*s.BK) qmem kv.key
  let vmem := storeTile "dv" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
    (tile.chunk*64) (tile.value*s.BV) kmem kv.value
  change final.mem=vmem.mem at memory
  refine ⟨final,ran,?_⟩
  intro r off outside
  calc
    _ = vmem.mem r off := congrFun (congrFun memory r) off
    _ = kmem.mem r off := PublicIO.storeTile_frame_full "dv" r _ _ _ _ _ _ _ off
      (fun i hi => outside.1 (2 : Fin 3) i hi)
    _ = qmem.mem r off := PublicIO.storeTile_frame_full "dk" r _ _ _ _ _ _ _ off
      (fun i hi => outside.1 (1 : Fin 3) i hi)
    _ = start.mem r off := PublicIO.storeTile_frame_full "dq" r _ _ _ _ _ _ _ off
      (fun i hi => outside.1 (0 : Fin 3) i hi)
    _ = _ := rfl

theorem forward_equivalent (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) :
    (PublicIO.forwardIO a p s tile).Equivalent (forwardVariant a p s tile) := by
  intro xs _ cap hb initial _
  have hs:=PublicIO.forward_schedule_safe a p s tile valid xs cap hb initial
  obtain ⟨final,ran,frame⟩:=forward_result a p s tile valid xs initial
  refine ⟨hs,?_,final,final,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe cap [PublicIO.forwardInvocation α s tile] initial at hs
    change ScheduleSafe cap [forwardInvocation s tile] initial
    rw [scheduleSafe_singleton] at hs ⊢
    exact (forward_safe algebra dots (config s 128) cap _).mp hs
  · change runSchedule [forwardInvocation s tile] initial=some final
    rw [runSchedule_singleton] at ran ⊢
    exact (forward_inv_run s tile initial).symm.trans ran
  · intro o j active
    rfl

theorem backward_equivalent (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) :
    (PublicIO.backwardIO a p s tile).Equivalent (backwardVariant a p s tile) := by
  intro xs _ cap hb initial _
  have hs:=PublicIO.backward_schedule_safe a p s tile valid xs cap hb initial
  obtain ⟨final,ran,frame⟩:=backward_result a p s tile valid xs initial
  refine ⟨hs,?_,final,final,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe cap [PublicIO.backwardInvocation α s tile] initial at hs
    change ScheduleSafe cap [backwardInvocation s tile] initial
    rw [scheduleSafe_singleton] at hs ⊢
    exact (shell_safe algebra dots (decodedBackwardConfig s (PublicIO.programIds s tile)) "dq" cap _).mp hs
  · change runSchedule [backwardInvocation s tile] initial=some final
    rw [runSchedule_singleton] at ran ⊢
    exact (backward_inv_run s tile initial).symm.trans ran
  · intro o j active
    rfl

#axiomsClean forward_equivalent
#axiomsClean backward_equivalent
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
