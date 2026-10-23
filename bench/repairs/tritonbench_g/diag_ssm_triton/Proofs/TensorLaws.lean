import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.LibraryContracts

namespace VeriTile.Bench.Repairs.DiagSsmTriton.TensorLaws
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts
set_option autoImplicit false

theorem Layout.transport {route : Route} {shape : List Nat} {before after : Tensor}
    (h : Layout route shape before) (same : SameShape before after) (dense : Dense after) :
    Layout route shape after := ⟨same.1.trans h.dtype, same.2.trans h.shape, dense⟩

theorem logical_frame (codec : Codec Word) (flip : Word → Word) (tensor : Tensor)
    (before after : Heap) (frame : Frame before after tensor.view.backing)
    (q : Nat) (hq : q < tensor.count) (k : Nat) (hk : k < tensor.route.components) :
    logicalWord codec flip tensor after q k = logicalWord codec flip tensor before q k := by
  unfold logicalWord
  rw [Frame.load codec tensor.view frame _ (tensor.bounded q hq k hk)]

theorem SameLogical.reframe {codec : Codec Word} {flip : Word → Word} {caller result : Tensor}
    {initial middleBefore middleAfter final : Heap}
    (same : SameLogical codec flip caller result middleBefore middleAfter)
    (before : Frame initial middleBefore caller.view.backing)
    (after : Frame middleAfter final result.view.backing) :
    SameLogical codec flip caller result initial final := by
  refine ⟨same.1, ?_⟩
  intro q hq k hk
  have hqr : q < result.count := by simpa [Tensor.count, same.1.2] using hq
  have hkr : k < result.route.components := by simpa [same.1.1] using hk
  rw [logical_frame codec flip result middleAfter final after q hqr k hkr,
    same.2 q hq k hk, logical_frame codec flip caller initial middleBefore before q hq k hk]

theorem normalize_any_success (codec : Codec Word) (ops : Operations Word) (api : Services)
    (_contracts : Contracts codec ops api) (role : Role) (tensor result : Tensor) (before after : State)
    (live : Resident before tensor)
    (run : Succeeded (Host.normalize api role tensor) before result after) :
    Extension before after ∧ SameLogical codec ops.flipSign tensor result before.heap after.heap ∧
      Resolved result ∧ Resident after result := by
  obtain ⟨rfl, rfl, flags⟩ := normalize_identity api role tensor result before after run
  exact ⟨Extension.refl _, ⟨⟨rfl, rfl⟩, fun _ _ _ _ => rfl⟩, flags, live⟩

theorem forward_sizes (route : Route) (d : Dimensions) (s x transition y : Tensor)
    (hs : Layout route [d.batch, d.dim] s) (hx : Layout route [d.length, d.batch, d.dim] x)
    (hl : Layout route [d.dim] transition) (hy : Layout route [d.length, d.batch, d.dim] y) :
    ForwardSizes route d ⟨s.view, x.view, transition.view, y.view⟩ := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa [Dimensions.width] using hs.words
  · simpa using hl.words
  · simpa [Dimensions.width] using hx.words
  · simpa [Dimensions.width] using hy.words

theorem backward_sizes (route : Route) (d : Dimensions) (s y transition gy gs gx gl : Tensor)
    (hs : Layout route [d.batch, d.dim] s) (hy : Layout route [d.length, d.batch, d.dim] y)
    (hl : Layout route [d.dim] transition) (hgy : Layout route [d.length, d.batch, d.dim] gy)
    (hgs : Layout route [d.batch, d.dim] gs) (hgx : Layout route [d.length, d.batch, d.dim] gx)
    (hgl : Layout route [d.batch, d.dim] gl) :
    BackwardSizes route d ⟨s.view, transition.view, y.view, gs.view, gx.view, gl.view, gy.view⟩ := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [Dimensions.width] using hs.words
  · simpa using hl.words
  · simpa [Dimensions.width] using hy.words
  · simpa [Dimensions.width] using hgs.words
  · simpa [Dimensions.width] using hgx.words
  · simpa [Dimensions.width] using hgl.words
  · simpa [Dimensions.width] using hgy.words

/-- The returned sum reads its actual input after allocator effects. The
allocator's ordinary live-input frame then identifies those words with the
pre-call gl_batch contents derived from the source grid. -/
theorem sum_returned (codec : Codec Word) (ops : Operations Word) (input result : Tensor)
    (before after : State) (reply : SumReply ops codec input result before after)
    (resident : Resident before input) (dense : Dense input)
    (contributions : Nat → Nat → Word)
    (actualInput : ∀ q, q < input.route.components * reply.dim → ∀ b, b < reply.batch →
      load codec before.heap (input.view.address (Reduction.inputWord input.route reply.dim q b)) = contributions q b)
    (q : Nat) (hq : q < input.route.components * reply.dim) :
    load codec after.heap (result.view.address q) =
      (reply.plan.tree q).evaluate ops.zero ops.add (contributions q) := by
  rw [reply.actual]
  apply Reduction.returned_from_contributions
  · intro q hq b hb
    have bound := Reduction.inputWord_bounded input.route reply.batch reply.dim q b hq hb
    have size : input.view.words = input.route.components * (reply.batch * reply.dim) := by
      simpa [Tensor.count, reply.inputShape] using dense.2.1
    rw [Frame.load codec input.view (reply.allocationFrame _ resident) _ (by simpa [size] using bound)]
    exact actualInput q hq b hb
  · exact hq

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.TensorLaws
