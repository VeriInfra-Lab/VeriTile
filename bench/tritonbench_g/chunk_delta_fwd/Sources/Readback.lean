import bench.tritonbench_g.chunk_delta_fwd.Sources.Memory
import bench.tritonbench_g.chunk_delta_fwd.Sources.Geometry
import bench.tritonbench_g.chunk_delta_fwd.Sources.StoreRange
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def memoryState (m : Memory α) : State α where
  mem := m
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem scatter_readback {shape : TileShape} (r : RegionName)
    (off : TileIndex shape → Nat) (value : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (m : Memory α)
    (i : TileIndex shape) (hi : active i)
    (hinj : ∀ a b, active a → active b → off a=off b → a=b) :
    scatter (TileShape.allIndices shape) r active off value m r (off i)=value i := by
  have h := masked_readback_active r off value active (memoryState m) i hi hinj
  rw [masked_scatter_memory] at h
  exact h

theorem history_read (M : Algebra α) (p : Parameters) (pid : PID) (c : Nat)
    (h : HValues p α) (m : Memory α) (i : HIndex p) (hc : c < p.NT) (hi : hActive p pid i) :
    writeHistory M p pid c h m "history" (hOffset p pid c i)=.mk .real (cast M (h i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
    (Geometry.history_collision p pid pid c c a b ha hb hc hc he).2.2)

theorem corrected_read (M : Algebra α) (p : Parameters) (pid : PID) (c b : Nat)
    (h : CIndex p → α) (m : Memory α) (i : CIndex p) (hi : cActive p pid c b i) :
    writeCorrected M p pid c b h m "corrected" (cOffset p pid c b i)=.mk .real (cast M (h i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a d ha hd he =>
    (Geometry.inner_collision p pid c b b a d ha hd he).2)

theorem final_read (M : Algebra α) (p : Parameters) (pid : PID)
    (h : HValues p α) (m : Memory α) (i : HIndex p) (hp : p.storeFinal=Bool.true)
    (hi : hActive p pid i) :
    writeFinal M p pid h m "final" (fOffset p pid i)=.mk .real (cast M (h i)) := by
  simp only [writeFinal,hp,if_true]
  exact scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
    (Geometry.final_collision p pid pid a b ha hb he).2)

theorem history_frame (M : Algebra α) (p : Parameters) (pid : PID) (c : Nat)
    (h : HValues p α) (m : Memory α) (r : RegionName) (o : Nat)
    (safe : r≠"history" ∨ ∀ i, hActive p pid i → o≠hOffset p pid c i) :
    writeHistory M p pid c h m r o=m r o := by
  apply scatter_frame
  rcases safe with hr|hi
  · exact Or.inl hr
  · exact Or.inr (fun i _ => hi i)

theorem corrected_frame (M : Algebra α) (p : Parameters) (pid : PID) (c b : Nat)
    (h : CIndex p → α) (m : Memory α) (r : RegionName) (o : Nat)
    (safe : r≠"corrected" ∨ ∀ i, cActive p pid c b i → o≠cOffset p pid c b i) :
    writeCorrected M p pid c b h m r o=m r o := by
  apply scatter_frame
  rcases safe with hr|hi
  · exact Or.inl hr
  · exact Or.inr (fun i _ => hi i)

theorem final_frame (M : Algebra α) (p : Parameters) (pid : PID)
    (h : HValues p α) (m : Memory α) (r : RegionName) (o : Nat)
    (safe : r≠"final" ∨ ∀ i, hActive p pid i → o≠fOffset p pid i) :
    writeFinal M p pid h m r o=m r o := by
  unfold writeFinal
  split
  · apply scatter_frame
    rcases safe with hr|hi
    · exact Or.inl hr
    · exact Or.inr (fun i _ => hi i)
  · rfl

theorem inner_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (c : Nat) (h : HValues p α) (m : Memory α) (r : RegionName) (o : Nat)
    (safe : r≠"corrected" ∨ ∀ b, b < innerCount p → ∀ i,
      cActive p pid c b i → o≠cOffset p pid c b i) :
    writeInner M D p pid input c h m r o=m r o := by
  unfold writeInner
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro b hb mem
  apply corrected_frame
  rcases safe with hr|hi
  · exact Or.inl hr
  · exact Or.inr (hi b hb)

theorem inner_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (c b : Nat) (h : HValues p α) (m : Memory α) (i : CIndex p)
    (hb : b < innerCount p) (hi : cActive p pid c b i) :
    writeInner M D p pid input c h m "corrected" (cOffset p pid c b i)=
      .mk .real (cast M (corrected M D p pid input c b h i)) := by
  apply foldRange_last _ (fun mem => mem "corrected" (cOffset p pid c b i))
    (innerCount p) b m _ hb
  · intro mem
    exact corrected_read M p pid c b _ mem i hi
  · intro d hd _ mem
    apply corrected_frame
    right
    intro j hj he
    have heq := (Geometry.inner_collision p pid c b d i j hi hj he).1
    omega

#axiomsClean scatter_readback
#axiomsClean inner_read
#axiomsClean inner_frame
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
