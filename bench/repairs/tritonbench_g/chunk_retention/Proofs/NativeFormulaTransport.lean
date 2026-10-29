import bench.repairs.tritonbench_g.chunk_retention.Proofs.FormulaCongruence
import bench.repairs.tritonbench_g.chunk_retention.Proofs.IOLayout

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket KernelStaging
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1200000

private abbrev nativeRead (ops : Ops α) (literal : ℝ→α) (s : State (Term α)) :=
  fun r off=>lower ops literal ((s.mem r off).read .real)

theorem native_history_eq (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (xs : Inputs α a.shape) (reads : Reads a xs (nativeRead ops literal s))
    (stage : a.stage=.forwardH) (head : s.pids 2<a.shape.B*a.shape.H) (n : Nat) :
    Reference.history ops a.shape (KernelNative.H.xs ops literal a.shape a.seed s) (s.pids 2) n=
      Reference.history ops a.shape (logicalInputs ops.zero a xs) (s.pids 2) n := by
  apply history_congr
  · exact reads.kAt_eq ops (Or.inl stage) _ head
  · exact reads.vAt_eq ops (Or.inl stage) _ head
  · funext e j
    unfold masked
    split_ifs with active
    · have h := reads.seed_eq ops.zero stage _ e j head active.1 active.2
      cases hs : a.seed <;> simpa [KernelNative.H.xs,KernelNative.inputs,readInputs,hs] using h
    · rfl

theorem native_reverse_eq (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (xs : Inputs α a.shape) (reads : Reads a xs (nativeRead ops literal s))
    (stage : a.stage=.backwardH) (head : s.pids 2<a.shape.B*a.shape.H) (n : Nat) :
    Reference.reverseCarry ops a.shape (KernelNative.BH.xs ops literal a.shape a.seed s)
      (literal a.scale) (s.pids 2) n=
      Reference.reverseCarry ops a.shape (logicalInputs ops.zero a xs) (literal a.scale) (s.pids 2) n := by
  apply reverse_congr
  · exact reads.qAt_eq ops (Or.inr (Or.inl stage)) _ head
  · exact reads.gAt_eq ops (Or.inl stage) _ head
  · funext e j
    unfold masked
    split_ifs with active
    · have h := reads.finalIncoming_eq ops.zero stage _ e j head active.1 active.2
      cases hs : a.seed <;> simpa [KernelNative.BH.xs,KernelNative.inputs,readInputs,hs] using h
    · rfl

theorem native_output_eq (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (xs : Inputs α a.shape) (reads : Reads a xs (nativeRead ops literal s))
    (stage : a.stage=.forwardO) (head : s.pids 2<a.shape.B*a.shape.H) (chunk : s.pids 1<a.shape.chunks) :
    Body.output ops a.shape (KernelNative.O.xs ops literal a.shape s) (literal a.scale)
      (s.pids 2) (s.pids 1) (KernelNative.O.history ops literal a.shape s)=
    Body.output ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (s.pids 2) (s.pids 1) (history ops.zero a.shape xs Bool.false (s.pids 2) (s.pids 1)) := by
  apply output_congr
  · exact reads.qAt_eq ops (Or.inl stage) _ head
  · exact reads.kAt_eq ops (Or.inr (Or.inl stage)) _ head
  · exact reads.vAt_eq ops (Or.inr (Or.inl stage)) _ head
  · exact reads.masked_history_eq ops Bool.false (Or.inl stage) _ _ head chunk

theorem native_query_eq (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (xs : Inputs α a.shape) (reads : Reads a xs (nativeRead ops literal s))
    (stage : a.stage=.backwardQKV) (head : s.pids 2<a.shape.B*a.shape.H) (chunk : s.pids 1<a.shape.chunks) :
    Body.dq ops a.shape (KernelNative.QKV.xs ops literal a.shape s) (literal a.scale)
      (s.pids 2) (s.pids 1) (KernelNative.QKV.history ops literal a.shape s "h")=
    Body.dq ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (s.pids 2) (s.pids 1) (history ops.zero a.shape xs Bool.false (s.pids 2) (s.pids 1)) := by
  apply query_congr
  · exact reads.kAt_eq ops (Or.inr (Or.inr stage)) _ head
  · exact reads.vAt_eq ops (Or.inr (Or.inr stage)) _ head
  · exact reads.gAt_eq ops (Or.inr stage) _ head
  · exact reads.masked_history_eq ops Bool.false (Or.inr stage) _ _ head chunk

theorem native_key_eq (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (xs : Inputs α a.shape) (reads : Reads a xs (nativeRead ops literal s))
    (stage : a.stage=.backwardQKV) (head : s.pids 2<a.shape.B*a.shape.H) (chunk : s.pids 1<a.shape.chunks) :
    Body.dk ops a.shape (KernelNative.QKV.xs ops literal a.shape s) (literal a.scale)
      (s.pids 2) (s.pids 1) (KernelNative.QKV.history ops literal a.shape s "dh")=
    Body.dk ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (s.pids 2) (s.pids 1) (history ops.zero a.shape xs Bool.true (s.pids 2) (s.pids 1)) := by
  apply key_congr
  · exact reads.qAt_eq ops (Or.inr (Or.inr stage)) _ head
  · exact reads.vAt_eq ops (Or.inr (Or.inr stage)) _ head
  · exact reads.gAt_eq ops (Or.inr stage) _ head
  · exact reads.masked_history_eq ops Bool.true stage _ _ head chunk

theorem native_value_eq (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (xs : Inputs α a.shape) (reads : Reads a xs (nativeRead ops literal s))
    (stage : a.stage=.backwardQKV) (head : s.pids 2<a.shape.B*a.shape.H) (chunk : s.pids 1<a.shape.chunks) :
    Body.dvPartial ops a.shape (KernelNative.QKV.xs ops literal a.shape s) (literal a.scale)
      (s.pids 0) (s.pids 2) (s.pids 1) (KernelNative.QKV.history ops literal a.shape s "dh")=
    Body.dvPartial ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (s.pids 0) (s.pids 2) (s.pids 1) (history ops.zero a.shape xs Bool.true (s.pids 2) (s.pids 1)) := by
  apply value_congr
  · exact reads.qAt_eq ops (Or.inr (Or.inr stage)) _ head
  · exact reads.kAt_eq ops (Or.inr (Or.inr stage)) _ head
  · exact reads.gAt_eq ops (Or.inr stage) _ head
  · exact reads.masked_history_eq ops Bool.true stage _ _ head chunk

/-- Original native packet observations agree with the independent finite
logical-array formula, under only the declared input binding and coordinates. -/
theorem native_formula_transport (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (legal : Legal a s.pids s.numPids) (xs : Inputs α a.shape)
    (binding : (nativeIO ops literal a s.pids s.numPids).InputBinding xs s)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a s.pids port j) :
    nativeTarget ops literal a s (observation a.shape port j)= (Cell.mk (outputs a.shape)[port].dtype (nativeFormula ops literal a s.pids xs port j)).read .real := by
  have reads := native_reads ops literal a s.pids s.numPids xs s binding
  rcases port with ⟨port,hp⟩
  have hport : port<7 := hp
  interval_cases port
  · exact congrArg (fun f=>ops.cast (f (s.pids 0*a.shape.BK+j.2.1.val) (s.pids 1*a.shape.BV+j.2.2.1.val)))
      (native_history_eq ops literal a s xs reads active.1 legal.1 j.1.val)
  · exact congrArg (fun f=>ops.cast (f (s.pids 0*a.shape.BK+j.1.val) (s.pids 1*a.shape.BV+j.2.1.val)))
      (native_history_eq ops literal a s xs reads active.1 legal.1 a.shape.chunks)
  · have hc : s.pids 1<a.shape.chunks := by
      have hl := legal
      simp only [Legal,active.1] at hl
      exact hl.2.2.2
    exact congrFun (congrFun (native_output_eq ops literal a s xs reads active.1 legal.1 hc) j.1.val)
      (s.pids 0*a.shape.BV+j.2.1.val)
  · exact congrArg (fun f=>ops.cast (f (s.pids 0*a.shape.BK+j.2.1.val) (s.pids 1*a.shape.BV+j.2.2.1.val)))
      (native_reverse_eq ops literal a s xs reads active.1 legal.1 j.1.val)
  · have hc : s.pids 1<a.shape.chunks := by
      have hl := legal
      simp only [Legal,active.1] at hl
      exact hl.2.2.2
    exact congrFun (congrFun (native_query_eq ops literal a s xs reads active.1 legal.1 hc) j.1.val)
      (s.pids 0*a.shape.BK+j.2.1.val)
  · have hc : s.pids 1<a.shape.chunks := by
      have hl := legal
      simp only [Legal,active.1] at hl
      exact hl.2.2.2
    exact congrFun (congrFun (native_key_eq ops literal a s xs reads active.1 legal.1 hc) j.1.val)
      (s.pids 0*a.shape.BK+j.2.1.val)
  · have hc : s.pids 1<a.shape.chunks := by
      have hl := legal
      simp only [Legal,active.1] at hl
      exact hl.2.2.2
    exact congrFun (congrFun (native_value_eq ops literal a s xs reads active.1 legal.1 hc) j.2.1.val)
      (j.1.val*a.shape.BV+j.2.2.1.val)

#axiomsClean native_formula_transport
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
