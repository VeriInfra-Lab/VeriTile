import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PairConnection

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 24000

/-- Same scalar ABI, replica arithmetic and local native expression profile. -/


theorem invocation_run (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (pid : PhysicalKernel.Program) (s : State (Replicas Word)) :
    (PublicIO.invocation ops.arithmetic Bool.true p pid).run s=(invocation ops p pid).run s := by
  exact (PairConnection.native_pair
    ops.arithmetic.model p).run _

theorem invocation_safe (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (pid : PhysicalKernel.Program) (bounds : RegionName → Nat) (s : State (Replicas Word)) :
    (PublicIO.invocation ops.arithmetic Bool.true p pid).Safe bounds s ↔ (invocation ops p pid).Safe bounds s := by
  exact (PairConnection.native_pair
    ops.arithmetic.model p).safe bounds _

theorem schedule_run (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (work : List PhysicalKernel.Program) (s : State (Replicas Word)) :
    runSchedule (work.map (PublicIO.invocation ops.arithmetic Bool.true p)) s=
      runSchedule (work.map (invocation ops p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    cases (invocation ops p pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (work : List PhysicalKernel.Program) (bounds : RegionName → Nat) (s : State (Replicas Word)) :
    ScheduleSafe bounds (work.map (PublicIO.invocation ops.arithmetic Bool.true p)) s ↔
      ScheduleSafe bounds (work.map (invocation ops p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Complete three-axis sparse grid, full CSR domain and all raw words. -/
theorem equivalent (ops : PhysicalKernel.NativePrimitives) (p : Parameters)
    (batches visits : Nat) (geometry : PublicIO.Geometry p) :
    (PublicIO.nativeIO ops p batches visits).EquivalentOn (candidate ops p batches visits)
      (PublicIO.Admissible p batches visits) := by
  intro vals admitted bounds hb initial hin
  obtain ⟨safe,final,ran,values,frame⟩ :=
    PublicIO.native_implements ops p batches visits geometry vals admitted bounds hb initial hin
  refine ⟨safe,(schedule_safe ops p (PhysicalKernel.programs p batches) bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run ops p (PhysicalKernel.programs p batches) initial).symm.trans ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairIO
