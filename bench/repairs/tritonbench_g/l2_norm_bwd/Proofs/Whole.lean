import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Context

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

/-- Actual same-call normalization/allocation, context and final reshape.
The returned descriptor may be a view object or an actual reshape copy. -/
structure CallPath {M α β : Type} (ops : Ops M α β) (red : Reduction α) (rt : Runtime M)
    (call : Call) (eps : α) (initial : World M) (output : View) (after : World M) where
  request : Request
  prepared : World M
  computed : World M
  normalize : normalize rt call initial=.ok request prepared
  context : contextual ops red rt request eps prepared=.ok () computed
  returned : rt.perform (.returnShape request.dx request.originalShape) computed=.ok output after

structure WholeSpec {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (call : Call) (eps : α) (initial : World M) (output : View) (after : World M) : Prop where
  path : Nonempty (CallPath ops red rt call eps initial output after)
  shape : output.shape=call.x.shape
  backed : output.backed
  size : output.numel=call.x.numel
  positive : 0<call.cols
  block : call.cols≤blockSize call.cols
  values : ∀ i,i<call.x.numel → logical ops neg after.memory output i=
    expected ops neg red call initial call.cols (blockSize call.cols) eps (i/call.cols) (i%call.cols)
  caller : Frame ops call.caller initial after
  device : after.device=initial.device

/-- Every successful actual return has the stated values and descriptor;
every error carries its actual prefix state and preserves all caller bytes. -/
theorem whole_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (eps : α) (initial : World M)
    (domain : InputDomain call initial) :
    Ensures (fun output after => WholeSpec ops neg red rt call eps initial output after)
      (fun _ after => Frame ops call.caller initial after) (run ops red rt call eps initial) := by
  unfold run
  have normalized := normalize_spec ops neg rt lib call initial domain
  have normalDevice := normalize_device ops neg rt lib call initial
  cases hn : normalize rt call initial with
  | error why failed =>
    rw [hn] at normalized
    exact normalized
  | ok q prepared =>
    rw [hn] at normalized normalDevice
    change Normalized ops neg call initial q prepared at normalized
    change prepared.device=initial.device at normalDevice
    simp only [Result.bind]
    have computed := contextual_spec ops neg red rt lib call initial prepared q eps normalized domain
    cases hc : contextual ops red rt q eps prepared with
    | error why failed =>
      rw [hc] at computed
      exact computed
    | ok unit s =>
      cases unit
      rw [hc] at computed
      change BodySpec ops neg red call initial q eps s ∧ s.device=prepared.device at computed
      simp only [Result.bind]
      have frame := lib.frame (.returnShape q.dx q.originalShape) s call.caller
        (fun a ha => computed.1.caller.live a (domain.live a ha))
      have device := lib.device (.returnShape q.dx q.originalShape) s
      cases hr : rt.perform (.returnShape q.dx q.originalShape) s with
      | error why failed =>
        simp only [hr,Result.state] at frame
        exact computed.1.caller.trans frame
      | ok output after =>
        simp only [hr,Result.state] at frame device
        obtain ⟨shape,copy⟩ := lib.returnShape _ _ _ _ _ hr
        have dxSize : q.dx.view.numel=call.x.numel := by
          rw [Matrix.numel,normalized.dxRows,normalized.dxCols,←Matrix.numel,normalized.xSize]
        refine ⟨⟨⟨q,prepared,s,hn,hc,hr⟩⟩,shape.trans normalized.shape,copy.backed,
          copy.size.trans dxSize,by rw [←normalized.cols];exact normalized.positive,
          by rw [←normalized.cols,←normalized.blockActual];exact normalized.block,?_,
          computed.1.caller.trans frame,device.trans (computed.2.trans normalDevice)⟩
        intro i hi
        have npos : 0<q.x.cols := normalized.positive
        have modBound : i%q.x.cols<q.x.cols := Nat.mod_lt i npos
        have rowBound : i/q.x.cols<q.x.rows := by
          apply (Nat.div_lt_iff_lt_mul npos).mpr
          rw [←Matrix.numel,normalized.xSize]
          exact hi
        have splitIndex : i=(i/q.x.cols)*q.x.cols+i%q.x.cols := by
          simpa [Nat.mul_comm] using (Nat.div_add_mod i q.x.cols).symm
        rw [copy.values i (by rw [dxSize];exact hi)]
        unfold VeriTile.Bench.Repairs.SwigluTriton.Memory.logical
        rw [normalized.dxPositive]
        simp only [Negation.encode,Bool.false_eq_true,if_false]
        conv_lhs => rw [splitIndex]
        rw [←normalized.dxCols,q.dx.logical_key _ _ (by rwa [normalized.dxCols]),
          normalized.dxCols,computed.1.values _ _ rowBound modBound,normalized.blockActual,normalized.cols]

#axiomsClean whole_spec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host
