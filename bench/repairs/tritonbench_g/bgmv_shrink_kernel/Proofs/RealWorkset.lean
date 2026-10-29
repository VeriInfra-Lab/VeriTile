import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.RealHost
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Linearization
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Physical

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.RealWorkset
open VeriTile.Triton Snapshot Geometry
noncomputable section

def initial (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0)
    (b : Fin h.batches) (n : Fin h.N) : ℝ :=
  (RealHost.initial h logical packed).readMem "Y" (outputOffset h b.val n.val)

def step (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0)
    (b : Fin h.batches) (n : Fin h.N) (split : Fin 64) (old : ℝ) : ℝ :=
  old+RealExecution.contribution h (RealHost.initial h logical packed)
    b.val (logical.indices b.val).toNat n.val split.val

abbrev Run (h : Call) (orders : Workset.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) :=
  Linearization.Run h orders logical.indices (initial h logical packed) (step h logical packed)

theorem payload_independent (h : Call) (a b : BlockState)
    (input : ∀ off,a.readMem "X" off=b.readMem "X" off)
    (weights : ∀ off,a.readMem "W" off=b.readMem "W" off) (batch slot n split : Nat) :
    RealExecution.contribution h a batch slot n split=RealExecution.contribution h b batch slot n split := by
  simp only [RealExecution.contribution,input,weights]

/-- Complete source execution at an arbitrary current output memory. Only the
immutable snapshot reads and local PID/metadata bindings are pinned. -/
theorem source_event (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0)
    (p : Program h) (u : BlockState)
    (batch : u.pids 1=p.batch.val) (split : u.pids 0=p.split.val)
    (valid : 0≤logical.indices p.batch.val) (active : logical.indices p.batch.val≠-1)
    (metadata : u.readMemValue .int "Indices" p.batch.val=logical.indices p.batch.val)
    (input : ∀ off,u.readMem "X" off=(RealHost.initial h logical packed).readMem "X" off)
    (weights : ∀ off,u.readMem "W" off=(RealHost.initial h logical packed).readMem "W" off) :
    ∃ v,exec (Kernels.source h).eraseDType.toAlgKernel u=some v ∧
      (∀ n : Fin h.N,v.readMem "Y" (outputOffset h p.batch.val n.val)=
        step h logical packed p.batch n p.split (u.readMem "Y" (outputOffset h p.batch.val n.val))) ∧
      (∀ r off,¬RealExecution.writes h p (logical.indices p.batch.val) r off → v.mem r off=u.mem r off) := by
  obtain ⟨v,execution,_,readback,frame⟩ := RealExecution.source_executes h p u batch split
    (logical.indices p.batch.val) (Or.inr valid) metadata
  refine ⟨v,execution,?_,frame⟩
  intro n
  rw [readback active n,payload_independent h u (RealHost.initial h logical packed) input weights]
  rfl

theorem readback (h : Call) (orders : Workset.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) (run : Run h orders logical packed)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨
      (0≤logical.indices b ∧ (logical.indices b).toNat<h.adapters))
    (b : Fin h.batches) (n : Fin h.N) :
    run.output b n=RealHost.math h logical b.val n.val := by
  rw [Linearization.Run.readback]
  rw [←RealHost.readback h orders logical packed valid b n]
  unfold RealHost.scratchOutput
  split_ifs
  · simp [initial,RealHost.initial,BlockState.readMem,MemCell.real,MemCell.of,MemCell.readAs]
  · rfl

def SourceWorkset (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0) : Prop :=
  ∀ (p : Program h) (u : BlockState),
    u.pids 1=p.batch.val → u.pids 0=p.split.val → logical.indices p.batch.val≠-1 →
    u.readMemValue .int "Indices" p.batch.val=logical.indices p.batch.val →
    (∀ off,u.readMem "X" off=(RealHost.initial h logical packed).readMem "X" off) →
    (∀ off,u.readMem "W" off=(RealHost.initial h logical packed).readMem "W" off) →
    ∃ v,exec (Kernels.source h).eraseDType.toAlgKernel u=some v ∧
      (∀ n : Fin h.N,v.readMem "Y" (outputOffset h p.batch.val n.val)=
        step h logical packed p.batch n p.split (u.readMem "Y" (outputOffset h p.batch.val n.val))) ∧
      (∀ r off,¬RealExecution.writes h p (logical.indices p.batch.val) r off → v.mem r off=u.mem r off)

theorem source_workset (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨ 0≤logical.indices b) :
    SourceWorkset h logical packed := by
  intro p u batch split active metadata input weights
  exact source_event h logical packed p u batch split
    ((valid p.batch.val p.batch.isLt).resolve_left active) active metadata input weights

def physical (h : Call) (orders : Workset.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) (run : Run h orders logical packed)
    (storage : Storage.Placement h) (before : Nat → ℝ) :=
  Physical.place storage logical.indices before run.output

theorem physical_readback (h : Call) (orders : Workset.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) (run : Run h orders logical packed)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨
      (0≤logical.indices b ∧ (logical.indices b).toNat<h.adapters))
    (storage : Storage.Placement h) (before : Nat → ℝ)
    (b : Fin h.batches) (n : Fin h.N) (active : logical.indices b.val≠-1) :
    physical h orders logical packed run storage before (Physical.outputAddress storage (b,n))=
      RealHost.math h logical b.val n.val := by
  rw [physical,Physical.readback storage logical.indices before run.output b n active]
  exact readback h orders logical packed run valid b n

#axiomsClean source_event
#axiomsClean readback
#axiomsClean physical_readback
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.RealWorkset
