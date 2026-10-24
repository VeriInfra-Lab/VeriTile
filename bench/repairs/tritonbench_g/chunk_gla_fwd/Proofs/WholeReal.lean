import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Producer
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.OutputLogical
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.WholeReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def outputShape (c : ProducerGeometry.Shape) (V BK BV : Nat) (hbv : 0<BV) : OutputGrid.Shape :=
  ⟨c.heads,c.T,c.K,V,c.BT,BK,BV,ProducerGeometry.bt_pos c,hbv⟩

theorem consume (c : OutputGrid.Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (hbk : 0<c.BK)
    (ha : ∀ (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT),
      s.readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
        Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val) :
    ∃ t, Nonempty (Kernel.GridLaunchedOrdinary (OutputGrid.kernel c scale) (OutputGrid.grid c) s t) ∧
      (∀ h row v, OutputGrid.returned c t h row v=
        Logical.output c.T c.K c.V c.BT (InterGrid.inputs s) scale h.val row.val v.val) ∧
      BlockState.WriteWithin (OutputGrid.footprint c) s t := by
  obtain ⟨t,he,ho,hf⟩ := OutputGrid.executes c scale s hu
  refine ⟨t,he,?_,hf⟩
  intro h row v
  rw [ho]
  exact OutputLogical.formula c (InterGrid.inputs s) (s.readMem "attention") scale h row v hbk
    (fun j hj => ha h row ⟨j,hj⟩)

/-- Constructed inter/direct-intra/consumer launches; all returned A and O
cells equal independent logical formulas. History and gate are shared inputs. -/
theorem direct (c : ProducerGeometry.Shape) (IBK : Nat) (hK : c.K≤IBK)
    (V OBK BV : Nat) (hobk : 0<OBK) (hbv : 0<BV) (scale : ℝ)
    (s : BlockState) (hu : s.undef=fun _ _ => 0) (hz : Producer.ZeroAttention c s) :
    ∃ a t, Nonempty (Producer.DirectRun c IBK scale s a) ∧
      Nonempty (Kernel.GridLaunchedOrdinary (OutputGrid.kernel (outputShape c V OBK BV hbv) scale)
        (OutputGrid.grid (outputShape c V OBK BV hbv)) a t) ∧
      (∀ (h : Fin c.heads) (row : Fin c.T) (j : Fin c.BT),
        t.readMem "attention" (h.val*c.T*c.BT+row.val*c.BT+j.val)=
          Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val row.val j.val) ∧
      (∀ h row v, OutputGrid.returned (outputShape c V OBK BV hbv) t h row v=
        Logical.output c.T c.K V c.BT (InterGrid.inputs s) scale h.val row.val v.val) ∧
      BlockState.WriteWithin (OutputGrid.footprint (outputShape c V OBK BV hbv)) a t := by
  let a := Producer.direct c IBK hK scale s hu
  have hin : InterGrid.inputs a=InterGrid.inputs s := Producer.direct_inputs c IBK hK scale s hu
  have ha : ∀ (h : Fin c.heads) (row : Fin c.T) (j : Fin c.BT),
      a.readMem "attention" (h.val*c.T*c.BT+row.val*c.BT+j.val)=
        Logical.attention c.T c.K c.BT (InterGrid.inputs a) scale h.val row.val j.val := by
    rw [hin]
    exact Producer.direct_readback c IBK hK scale s hu hz
  obtain ⟨t,he,ho,hf⟩ := consume (outputShape c V OBK BV hbv) scale a hu hobk ha
  refine ⟨a,t,⟨Producer.direct_runs c IBK hK scale s hu⟩,he,?_,?_,hf⟩
  · intro h row j
    unfold BlockState.readMem
    rw [← hf "attention" _ (by simp [OutputGrid.footprint])]
    exact Producer.direct_readback c IBK hK scale s hu hz h row j
  · intro h row v
    simpa only [hin] using ho h row v

/-- Constructed inter/split/merge/consumer launches, including all feature,
time and value tails and all zero cells of the rectangular A result. -/
theorem large (c : ProducerGeometry.Shape) (V OBK BV : Nat) (hobk : 0<OBK) (hbv : 0<BV) (scale : ℝ)
    (s : BlockState) (hu : s.undef=fun _ _ => 0) (hz : Producer.ZeroAttention c s) (hs : Producer.ZeroScratch c s) :
    ∃ a t, Nonempty (Producer.LargeRun c scale s a) ∧
      Nonempty (Kernel.GridLaunchedOrdinary (OutputGrid.kernel (outputShape c V OBK BV hbv) scale)
        (OutputGrid.grid (outputShape c V OBK BV hbv)) a t) ∧
      (∀ (h : Fin c.heads) (row : Fin c.T) (j : Fin c.BT),
        t.readMem "attention" (h.val*c.T*c.BT+row.val*c.BT+j.val)=
          Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val row.val j.val) ∧
      (∀ h row v, OutputGrid.returned (outputShape c V OBK BV hbv) t h row v=
        Logical.output c.T c.K V c.BT (InterGrid.inputs s) scale h.val row.val v.val) ∧
      BlockState.WriteWithin (OutputGrid.footprint (outputShape c V OBK BV hbv)) a t := by
  let a := Producer.large c scale s hu
  have hin : InterGrid.inputs a=InterGrid.inputs s := Producer.large_inputs c scale s hu
  have ha : ∀ (h : Fin c.heads) (row : Fin c.T) (j : Fin c.BT),
      a.readMem "attention" (h.val*c.T*c.BT+row.val*c.BT+j.val)=
        Logical.attention c.T c.K c.BT (InterGrid.inputs a) scale h.val row.val j.val := by
    rw [hin]
    exact Producer.large_readback c scale s hu hz hs
  obtain ⟨t,he,ho,hf⟩ := consume (outputShape c V OBK BV hbv) scale a hu hobk ha
  refine ⟨a,t,⟨Producer.large_runs c scale s hu⟩,he,?_,?_,hf⟩
  · intro h row j
    unfold BlockState.readMem
    rw [← hf "attention" _ (by simp [OutputGrid.footprint])]
    exact Producer.large_readback c scale s hu hz hs h row j
  · intro h row v
    simpa only [hin] using ho h row v

#axiomsClean direct
#axiomsClean large
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.WholeReal
