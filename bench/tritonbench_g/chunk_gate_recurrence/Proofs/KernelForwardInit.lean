import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelInitial
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelForwardLoop

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelForwardInit
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelMemory KernelForwardLoop
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem fold_memory {ι : Type} (r : RegionName) (off : ι → Nat) (value : ι → Cell α)
    (indices : List ι) (u v : State α) (same : u.mem=v.mem) :
    (indices.foldl (fun w i => w.write r (off i) (value i)) u).mem=
      (indices.foldl (fun w i => w.write r (off i) (value i)) v).mem := by
  induction indices generalizing u v with
  | nil => exact same
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [State.write,same]

theorem memory (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    (KernelPrefix.forward M c s seed u).val.mem=
      (scatter "O" .real (fun e j => address c p 0 e.val j.val)
        (fun i => M.cast (some .fp32) .real .real (KernelInitial.initialCarry M c s seed p u i)) u).mem := by
  cases seed <;>
    simp only [KernelPrefix.forward,State.setReg_mem]
  all_goals
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
      hh,hk,hv,KernelProgram.scatter,KernelInitial.initialCarry,KernelProgram.raw,
      KernelProgram.address,layout,launch,TritonBenchG.ChunkGateRecurrence.RealProgram.address,
      seedAddress,toFloat,ofFloat]
    apply fold_memory
    rfl

theorem holds (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    Holds M c s seed p u 0 (KernelPrefix.forward M c s seed u).val := by
  obtain ⟨hs,ho,hd,ha⟩ := KernelInitial.forward_registers M c s seed p u hh hk hv
  have hm := memory M c s seed p u hh hk hv
  refine ⟨Nat.zero_le _,hs,ho,hd,?_,?_,?_,?_,?_⟩
  · simpa [carry,KernelReference.forward,KernelReference.initial,KernelReference.read,
      KernelReference.zero,KernelInitial.initialCarry] using ha
  · intro off
    rw [hm]
    exact scatter_other _ _ _ _ _ _ _ (by decide)
  · intro off
    rw [hm]
    exact scatter_other _ _ _ _ _ _ _ (by decide)
  · intro t ht e j
    have eq : t=0 := by omega
    subst t
    rw [hm,scatter_cell]
    rfl
  · intro r off outside
    rw [hm]
    exact forward_frame c p ⟨0,nonempty⟩ _ u r off outside

theorem source_correct (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    ∃ v, FP.Structural.exec (engine M) (source .forward c s seed) u=some v ∧
      (∀ (t : Fin c.N) (e : Fin 64) (j : Fin 16),
        v.mem "O" (address c p t.val e.val j.val)=
          KernelReference.stored M .real (carry M c s seed p u t.val (e,j,PUnit.unit))) ∧
      Frame .forward c p u v := by
  obtain ⟨v,he,inv⟩ := KernelForwardLoop.executes M c s seed p u 0
    (KernelPrefix.forward M c s seed u).val (holds M c s seed p u nonempty hh hk hv)
  refine ⟨v,?_,fun t e j => inv.output t.val (by omega) e j,inv.frame⟩
  change run (engine M) (source .forward c s seed).surfaceBody u=some v
  rw [source_body,List.append_assoc,run_append,(KernelPrefix.forward M c s seed u).property]
  simpa [KernelProgram.loop,KernelOps.forward_body,after,parts,source,Kernels.forward,
    ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardFamily,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardSumLeftFamily,VeriTile.Triton.Source.dynamicRange?,run,step,FP.Structural.evalOp] using he

#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelForwardInit
