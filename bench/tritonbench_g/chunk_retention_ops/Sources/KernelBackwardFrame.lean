import bench.tritonbench_g.chunk_retention_ops.Sources.KernelBackwardReadback
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset scatter_frame)
namespace Geo
export VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry (StatePID history_bound final_bound chunk_lt)
end Geo
noncomputable section
variable {α : Type} [Inhabited α]

def Exterior (p : Parameters) (r : RegionName) (off : Nat) : Prop :=
  (r="dhistory" → p.B*p.H*p.NT*p.K*p.V≤off) ∧
  (r="initialGradient" → p.B*p.H*p.K*p.V≤off)

theorem history_frame (M : Algebra α) (p : Parameters) (pid : PID) (hp : Geo.StatePID p pid)
    (n : Nat) (hn : n<p.NT) (carry : Carry p α) (m : Memory α)
    (r : RegionName) (off : Nat) (h : Exterior p r off) :
    writeHistory M p pid n carry m r off=m r off := by
  by_cases hr : r="dhistory"
  · subst r
    apply scatter_frame
    right
    intro i _ hi he
    have hb:=Geo.history_bound p pid (time p n) i hp (Geo.chunk_lt p Bool.true n hn) hi
    have ho:=h.1 rfl
    omega
  · exact scatter_frame _ _ _ _ _ _ _ _ (Or.inl hr)

theorem initial_frame (M : Algebra α) (p : Parameters) (pid : PID) (hp : Geo.StatePID p pid)
    (initial : Bool) (carry : Carry p α) (m : Memory α)
    (r : RegionName) (off : Nat) (h : Exterior p r off) :
    writeInitial M p pid initial carry m r off=m r off := by
  cases initial
  · rfl
  · by_cases hr : r="initialGradient"
    · subst r
      apply scatter_frame
      right
      intro i _ hi he
      have hb:=Geo.final_bound p pid i hp hi
      have ho:=h.2 rfl
      omega
    · exact scatter_frame _ _ _ _ _ _ _ _ (Or.inl hr)

theorem memory_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (hp : Geo.StatePID p pid) (input m : Memory α) (scale : ℝ) (terminal initial : Bool)
    (r : RegionName) (off : Nat) (h : Exterior p r off) :
    memory M D p pid input m scale terminal initial r off=m r off := by
  unfold memory
  rw [initial_frame M p pid hp initial _ _ r off h]
  apply foldRange_observe _ (fun mem : Memory α=>mem r off) p.NT m
  intro n hn mem
  exact history_frame M p pid hp n hn _ mem r off h

#axiomsClean memory_frame
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
