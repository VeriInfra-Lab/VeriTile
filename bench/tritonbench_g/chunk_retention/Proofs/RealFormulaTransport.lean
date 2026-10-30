import bench.tritonbench_g.chunk_retention.Proofs.NativeFormulaTransport
import bench.tritonbench_g.chunk_retention.Sources.RealReference

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket KernelStaging
noncomputable section
set_option maxHeartbeats 1500000

private theorem history_real (c : Shape) (xs : Whole001.Inputs ℝ) (h n e j : Nat)
    (hn : n<c.chunks) (he : e<c.K) (hj : j<c.V) :
    Reference.history realOps c xs h n e j=MathTarget.state c xs Bool.true h (n*64) e j :=
  (RealScalar.history_scalar c xs h n e j (Nat.le_of_lt hn) he hj).trans
    (ScalarSource.history_correct c xs Bool.true h n e j hn)

private theorem reverse_real (c : Shape) (xs : Whole001.Inputs ℝ) (scale : ℝ) (h n e j : Nat)
    (hn : n≤c.chunks) (he : e<c.K) (hj : j<c.V) :
    Reference.reverseCarry realOps c xs scale h n e j=
      MathTarget.future c xs scale h (c.boundary (c.chunks-n)) e j :=
  (RealScalar.reverse_scalar c xs scale h n e j hn he hj).trans
    (ScalarSource.reverse_carry_correct c xs scale h n hn e j)

/-- The scalar graph interpreted over Reals is the independent mathematical
formula. This is an exact mathematical lemma, not an experimental assumption. -/
theorem native_formula_real (a : Call) (p grid : Nat→Nat) (legal : Legal a p grid)
    (xs : Inputs ℝ a.shape) (port : Fin (outputs a.shape).length)
    (j : TileIndex (outputs a.shape)[port].shape) (active : outputActive a p port j) :
    nativeFormula realOps id a p xs port j=realFormula a p xs port j := by
  rcases port with ⟨port,hp⟩
  have hport : port<7 := hp
  interval_cases port
  · exact history_real _ _ _ _ _ _ j.1.isLt active.2.1 active.2.2
  · exact RealScalar.reference_final _ _ _ _ _ active.2.2.1 active.2.2.2
  · exact RealScalar.output_real _ _ _ _ _ _ _ _ active.2.1 j.1.isLt active.2.2
  · exact reverse_real _ _ _ _ _ _ _ (Nat.le_of_lt j.1.isLt) active.2.1 active.2.2
  · exact RealScalar.query_real _ _ _ _ _ _ _ _ active.2.1 j.1.isLt active.2.2
  · have hc : p 1<a.shape.chunks := by
      have hl := legal
      simp only [Legal,active.1] at hl
      exact hl.2.2.2
    exact RealScalar.key_real _ _ _ _ _ _ _ _ hc
      ((RealScalar.time_lane a.shape (p 1) j.1.val hc j.1.isLt).mp active.2.1) active.2.2
  · have hc : p 1<a.shape.chunks := by
      have hl := legal
      simp only [Legal,active.1] at hl
      exact hl.2.2.2
    exact RealScalar.value_real _ _ _ _ _ _ _ _ _ hc
      ((RealScalar.time_lane a.shape (p 1) j.2.1.val hc j.2.1.isLt).mp active.2.1) active.2.2

/-- Reification here transports only a formula's input values. It is not a new
source, execution semantics, or additional condition on actual tensor memory. -/
theorem packet_native_real (a : Call) (s : State ℝ) (legal : KernelPacket.Legal a s)
    (o : Observation a.shape) (active : Active a s o) :
    nativeTarget realOps id a (KernelNative.QKV.reify s) o=realTarget a s o := by
  cases o with
  | history n i =>
    change Reference.history realOps a.shape (KernelReal.H.xs a.shape a.seed s)
      (s.pids 2) n (ForwardHSource.feature a.shape s i.1.val) (ForwardHSource.column a.shape s i.2.1.val)=_
    exact history_real _ _ _ _ _ _ active.2.1 active.2.2.1 active.2.2.2
  | finalState i =>
    change Reference.finalState realOps a.shape (KernelReal.H.xs a.shape a.seed s)
      (s.pids 2) (ForwardHSource.feature a.shape s i.1.val) (ForwardHSource.column a.shape s i.2.1.val)=_
    exact RealScalar.reference_final _ _ _ _ _ active.2.2.1 active.2.2.2
  | output i =>
    change Body.output realOps a.shape (KernelReal.O.xs a.shape s) a.scale (s.pids 2) (s.pids 1)
      (KernelReal.O.history a.shape s) i.1.val (ForwardOSource.column a.shape s i.2.1.val)=_
    exact RealScalar.output_real _ _ _ _ _ _ _ _ active.2.1 i.1.isLt active.2.2
  | dhistory n i =>
    change Reference.reverseCarry realOps a.shape (KernelReal.BH.xs a.shape a.seed s) a.scale
      (s.pids 2) n (BackwardHSource.feature a.shape s i.1.val) (BackwardHSource.column a.shape s i.2.1.val)=_
    exact reverse_real _ _ _ _ _ _ _ (Nat.le_of_lt active.2.1) active.2.2.1 active.2.2.2
  | query i =>
    change Body.dq realOps a.shape (KernelReal.QKV.xs a.shape s) a.scale (s.pids 2) (s.pids 1)
      (KernelReal.QKV.history a.shape s "h") i.1.val (BackwardQKVSource.feature a.shape s i.2.1.val)=_
    exact RealScalar.query_real _ _ _ _ _ _ _ _ active.2.1 i.1.isLt active.2.2
  | key i =>
    change Body.dk realOps a.shape (KernelReal.QKV.xs a.shape s) a.scale (s.pids 2) (s.pids 1)
      (KernelReal.QKV.history a.shape s "dh") i.1.val (BackwardQKVSource.feature a.shape s i.2.1.val)=_
    have hc : s.pids 1<a.shape.chunks := by
      have hl := legal
      simp only [KernelPacket.Legal,active.1] at hl
      exact hl.2.2.2
    exact RealScalar.key_real _ _ _ _ _ _ _ _ hc
      ((RealScalar.time_lane a.shape (s.pids 1) i.1.val hc i.1.isLt).mp active.2.1) active.2.2
  | valuePart f i =>
    change Body.dvPartial realOps a.shape (KernelReal.QKV.xs a.shape s) a.scale
      (s.pids 0) (s.pids 2) (s.pids 1) (KernelReal.QKV.history a.shape s "dh")
      i.1.val (BackwardQKVSource.column a.shape f i.2.1.val)=_
    have hc : s.pids 1<a.shape.chunks := by
      have hl := legal
      simp only [KernelPacket.Legal,active.1] at hl
      exact hl.2.2.2
    exact RealScalar.value_real _ _ _ _ _ _ _ _ _ hc
      ((RealScalar.time_lane a.shape (s.pids 1) i.1.val hc i.1.isLt).mp active.2.2.1) active.2.2.2

theorem reify_input_binding (a : Call) (s : State ℝ) (xs : Inputs ℝ a.shape)
    (binding : (realIO a s.pids s.numPids).InputBinding xs s) :
    (nativeIO realOps id a s.pids s.numPids).InputBinding xs (KernelNative.QKV.reify s) := by
  intro port j enabled
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  all_goals
    apply congrArg (Term.input _ _)
    exact binding _ j enabled

theorem real_formula_transport (a : Call) (s : State ℝ) (legal : Legal a s.pids s.numPids)
    (xs : Inputs ℝ a.shape) (binding : (realIO a s.pids s.numPids).InputBinding xs s)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a s.pids port j) :
    realTarget a s (observation a.shape port j)=
      (Cell.mk (outputs a.shape)[port].dtype (realFormula a s.pids xs port j)).read .real := by
  rw [← packet_native_real a s legal _ ((observation_active a s port j).mpr active)]
  have ht := native_formula_transport realOps id a (KernelNative.QKV.reify s) legal xs
    (reify_input_binding a s xs binding) port j active
  change nativeTarget realOps id a (KernelNative.QKV.reify s) (observation a.shape port j)=
    (Cell.mk (outputs a.shape)[port].dtype (nativeFormula realOps id a s.pids xs port j)).read .real at ht
  rw [native_formula_real a s.pids s.numPids legal xs port j active] at ht
  exact ht

#axiomsClean native_formula_real
#axiomsClean packet_native_real
#axiomsClean real_formula_transport
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
