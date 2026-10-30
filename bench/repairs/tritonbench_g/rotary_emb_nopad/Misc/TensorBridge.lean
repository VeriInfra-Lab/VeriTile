import bench.repairs.tritonbench_g.rotary_emb_nopad.Correct
import bench.repairs.tritonbench_g.rotary_emb_nopad.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.RealBinding

/-! Optional checked Tensor/byte applications of the cached and uncached
primary source specifications. These declarations are ordinary theorems. -/
namespace VeriTile.Bench.Repairs.RotaryEmbNopad.TensorBridge
open VeriTile.Triton FP.Structural
open Checked Correct FPEquiv
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000

export TypedBinding (positions c_destination)
/-- Complete uncached typed execution on arbitrary Real payloads. The
independent split-half formula is a*c-b*s / a*s+b*c, using the entry-memory
coefficient pair. GQA leaders own K; the complete memory frame is retained. -/
theorem rotary_emb_nopad_uncached_correct (rest : Algebra ℝ) (c : Checked.Call)
    (index : Indices) (accepted : Checked.check c index=Bool.true)
    (_uncached : c.request.cached=Bool.false) (s : State ℝ) :
    Spec.Real (∃ t,exec (Typed.algebra rest realOps) (Kernels.uncached (Typed.uLayout c.request)) s=some t ∧
      t.mem=(Typed.uTarget realOps c.request s).mem ∧
      (∀ key odd i,Typed.ULive c.request s key i →
        t.mem (Typed.uRegion (Typed.uLayout c.request) key) (Typed.uOffset (Typed.uLayout c.request) s key odd i)=
          Typed.uValue realOps (Typed.uLayout c.request) s s key odd i) ∧
      (∀ r o,Typed.UOutside c.request s r o → t.mem r o=s.mem r o)) :=
  Typed.uncached_successful rest realOps c index accepted s

/-- Complete cached typed execution, including padded-head early return,
checked signed page/slot arithmetic, cache stores before Q, and unchanged K.
Metadata observations are separate from unrestricted Real payloads. -/
theorem rotary_emb_nopad_cached_correct (rest : Algebra ℝ) (c : Checked.Call)
    (index : Indices) (accepted : Checked.check c index=Bool.true)
    (cached : c.request.cached=Bool.true) (s : State ℝ)
    (observed : Typed.MetadataMatches c.request index s)
    (token : s.pids 1<c.request.tokens) (nonempty : 0<c.request.half) :
    Spec.Real (∃ t,exec (Typed.algebra rest realOps) (Kernels.cached (Typed.cLayout c.request)) s=some t ∧
      t.mem=(Typed.cReference realOps (Typed.cLayout c.request) s).mem ∧
      (∀ key odd i,s.pids 0<c.request.qHeads → Typed.cActive (Typed.cLayout c.request) s key i →
        t.mem (Typed.cRegion (Typed.cLayout c.request) key) (Typed.cOffset (Typed.cLayout c.request) s key odd i)=
          Typed.cValue realOps (Typed.cLayout c.request) s key odd i) ∧
      (∀ r o,Typed.COutside c.request s r o → t.mem r o=s.mem r o) ∧
      (∀ o,t.mem "K" o=s.mem "K" o)) :=
  Typed.cached_successful rest realOps c index accepted cached s observed token nonempty



/-- Successful complete-source equivalence to the independent selected native
FP32 graph. Uncached multiplication order and FMA contraction are preserved.
There is no historical optimization pair for this variant. Primitive semantics
are the model boundary, with no numerical obligations or finite-word guards. -/
theorem rotary_emb_nopad_uncached_fp32_equiv (rest : Algebra Native.Node)
    (p : Native.Instructions) (c : Checked.Call) (index : Indices)
    (accepted : Checked.check c index=Bool.true) (_uncached : c.request.cached=Bool.false)
    (s : State Native.Node) :
    ∃ t,exec (Typed.algebra rest (Native.ops p Bool.false)) (Kernels.uncached (Typed.uLayout c.request)) s=some t ∧
      t.mem=(Typed.uGraphTarget p c.request s).mem ∧
      (∀ key odd i,Typed.ULive c.request s key i →
        t.mem (Typed.uRegion (Typed.uLayout c.request) key) (Typed.uOffset (Typed.uLayout c.request) s key odd i)=
          Typed.uGraphValue p c.request s key odd i) ∧
      (∀ r o,Typed.UOutside c.request s r o → t.mem r o=s.mem r o) := by
  simpa only [Typed.u_target_graph,Typed.u_value_graph] using
    Typed.uncached_successful rest (Native.ops p Bool.false) c index accepted s

/-- The cached kernel has a distinct selected native graph and store order.
The statement includes every typed cell, early return, and unchanged input K. -/
theorem rotary_emb_nopad_cached_fp32_equiv (rest : Algebra Native.Node)
    (p : Native.Instructions) (c : Checked.Call) (index : Indices)
    (accepted : Checked.check c index=Bool.true) (cached : c.request.cached=Bool.true)
    (s : State Native.Node) (observed : Typed.MetadataMatches c.request index s)
    (token : s.pids 1<c.request.tokens) (nonempty : 0<c.request.half) :
    ∃ t,exec (Typed.algebra rest (Native.ops p Bool.true)) (Kernels.cached (Typed.cLayout c.request)) s=some t ∧
      t.mem=(Typed.cGraphTarget p c.request s).mem ∧
      (∀ key odd i,s.pids 0<c.request.qHeads → Typed.cActive (Typed.cLayout c.request) s key i →
        t.mem (Typed.cRegion (Typed.cLayout c.request) key) (Typed.cOffset (Typed.cLayout c.request) s key odd i)=
          Typed.cGraphValue p c.request s key odd i) ∧
      (∀ r o,Typed.COutside c.request s r o → t.mem r o=s.mem r o) ∧
      (∀ o,t.mem "K" o=s.mem "K" o) := by
  simpa only [Typed.c_target_graph,Typed.c_value_graph] using
    Typed.cached_successful rest (Native.ops p Bool.true) c index accepted cached s observed token nonempty

/-- Bind the uncached Real source to actual Tensor logical elements and the
current caller bytes through an explicit total word interpretation. -/
theorem rotary_emb_nopad_checked_uncached_real (rest : Algebra ℝ)
    (c : Checked.Call) (m : Physical.Bytes)
    (accepted : Checked.check c (Physical.indices c.request m)=Bool.true)
    (uncached : c.request.cached=Bool.false) (objects : TypedBinding.Objects c)
    (decode : Native.Word → ℝ) (surrounding : State ℝ)
    (head : surrounding.pids 0<c.request.qHeads) :
    Spec.Real (let s := RealBinding.state c.request decode m surrounding
      ∃ t,exec (Typed.algebra rest realOps) (Kernels.uncached (Typed.uLayout c.request)) s=some t ∧
        t.mem=(Typed.uTarget realOps c.request s).mem ∧
        (∀ i o,o<(c.tensor (TypedBinding.payloadRole (RealBinding.inputRole i))).shape.prod →
          decode (Physical.load m (.other c.request.device)
            (Checked.base c.request (TypedBinding.payloadRole (RealBinding.inputRole i))+o))=
              RealBinding.logical c objects decode m i o) ∧
        (∀ key odd i,Typed.ULive c.request surrounding key i →
          t.mem (Typed.uRegion (Typed.uLayout c.request) key) (Typed.uOffset (Typed.uLayout c.request) s key odd i)=
            .mk .real (RealBinding.logicalPair c objects decode m
              (TypedBinding.uWork c.request surrounding key i) odd)) ∧
        (∀ r o,Typed.UOutside c.request s r o → t.mem r o=s.mem r o)) := by
  dsimp only [VeriTile.Spec.Real]
  obtain ⟨t,source,memory,values,frame⟩ := rotary_emb_nopad_uncached_correct rest c _ accepted uncached
    (RealBinding.state c.request decode m surrounding)
  refine ⟨t,source,memory,RealBinding.direct c objects decode m accepted,?_,frame⟩
  intro key odd i live
  exact (values key odd i live).trans
    (RealBinding.u_value_logical c objects decode m accepted surrounding head key odd i live)

/-- The cached Real statement observes signed metadata directly from caller
bytes, derives the logical input formula, and retains every input K cell and
the complete frame. No payload or numerical conformance guard is added. -/
theorem rotary_emb_nopad_checked_cached_real (rest : Algebra ℝ)
    (c : Checked.Call) (m : Physical.Bytes)
    (accepted : Checked.check c (Physical.indices c.request m)=Bool.true)
    (cached : c.request.cached=Bool.true) (objects : TypedBinding.Objects c)
    (decode : Native.Word → ℝ) (surrounding : State ℝ)
    (token : surrounding.pids 1<c.request.tokens) (nonempty : 0<c.request.half) :
    Spec.Real (let s := RealBinding.state c.request decode m surrounding
      ∃ t,exec (Typed.algebra rest realOps) (Kernels.cached (Typed.cLayout c.request)) s=some t ∧
        t.mem=(Typed.cReference realOps (Typed.cLayout c.request) s).mem ∧
        (∀ i o,o<(c.tensor (TypedBinding.payloadRole (RealBinding.inputRole i))).shape.prod →
          decode (Physical.load m (.other c.request.device)
            (Checked.base c.request (TypedBinding.payloadRole (RealBinding.inputRole i))+o))=
              RealBinding.logical c objects decode m i o) ∧
        (∀ key odd i,surrounding.pids 0<c.request.qHeads → Typed.cActive (Typed.cLayout c.request) surrounding key i →
          t.mem (Typed.cRegion (Typed.cLayout c.request) key) (Typed.cOffset (Typed.cLayout c.request) s key odd i)=
            .mk .real (RealBinding.logicalPair c objects decode m
              (TypedBinding.cWork c.request surrounding key i) odd)) ∧
        (∀ r o,Typed.COutside c.request s r o → t.mem r o=s.mem r o) ∧
        (∀ o,t.mem "K" o=s.mem "K" o)) := by
  dsimp only [VeriTile.Spec.Real]
  obtain ⟨t,source,memory,values,frame,keys⟩ := rotary_emb_nopad_cached_correct rest c _ accepted cached
    (RealBinding.state c.request decode m surrounding)
    (RealBinding.metadata_matches c.request decode m surrounding) token nonempty
  refine ⟨t,source,memory,RealBinding.direct c objects decode m accepted,?_,frame,keys⟩
  intro key odd i head live
  exact (values key odd i head live).trans
    (RealBinding.c_value_logical c objects decode m accepted surrounding token head key odd i live)

theorem rotary_emb_nopad_checked_uncached_fp32 (rest : Algebra Native.Node)
    (p : Native.Instructions) (c : Checked.Call) (m : Physical.Bytes)
    (accepted : Checked.check c (Physical.indices c.request m)=Bool.true)
    (uncached : c.request.cached=Bool.false) (objects : TypedBinding.Objects c)
    (surrounding : State Native.Node) (head : surrounding.pids 0<c.request.qHeads) :
    let s := TypedBinding.state c.request m surrounding
    ∃ t,exec (Typed.algebra rest (Native.ops p Bool.false)) (Kernels.uncached (Typed.uLayout c.request)) s=some t ∧
      t.mem=(Typed.uGraphTarget p c.request s).mem ∧
      (∀ r o,Typed.UOutside c.request s r o → t.mem r o=s.mem r o) ∧
      (∀ i pi o,o<(c.tensor (TypedBinding.payloadRole i)).shape.prod →
        Physical.load m (.other c.request.device) (Checked.base c.request (TypedBinding.payloadRole i)+o)=
          TypedBinding.logical c objects m i pi o) ∧
      (∀ key odd i,Typed.ULive c.request surrounding key i →
        Physical.load (Physical.native p c m) (.other c.request.device)
          (destination c.request (Physical.indices c.request m) (TypedBinding.uWork c.request surrounding key i) odd)=
          ((t.mem (Typed.uRegion (Typed.uLayout c.request) key)
            (Typed.uOffset (Typed.uLayout c.request) surrounding key odd i)).read .real).value) ∧
      (∀ r address,(∀ w,c.request.Active w → ¬Writes c.request (Physical.indices c.request m) w r (address/4)) →
        Physical.native p c m r address=m r address) := by
  dsimp only
  obtain ⟨t,source,memory,_,frame⟩ := rotary_emb_nopad_uncached_fp32_equiv rest p c _ accepted uncached
    (TypedBinding.state c.request m surrounding)
  have same : t=(Typed.uncachedExecution rest (Native.ops p Bool.false) (Typed.uLayout c.request)
      (TypedBinding.state c.request m surrounding)).val :=
    Option.some.inj (source.symm.trans (Typed.uncachedExecution rest (Native.ops p Bool.false)
      (Typed.uLayout c.request) (TypedBinding.state c.request m surrounding)).property)
  refine ⟨t,source,memory,frame,TypedBinding.direct c objects m accepted,?_,Physical.native_frame p c m accepted⟩
  subst t
  exact TypedBinding.u_physical rest p c m accepted uncached surrounding head

theorem rotary_emb_nopad_checked_cached_fp32 (rest : Algebra Native.Node)
    (p : Native.Instructions) (c : Checked.Call) (m : Physical.Bytes)
    (accepted : Checked.check c (Physical.indices c.request m)=Bool.true)
    (cached : c.request.cached=Bool.true) (objects : TypedBinding.Objects c)
    (surrounding : State Native.Node) (token : surrounding.pids 1<c.request.tokens)
    (nonempty : 0<c.request.half) :
    let s := TypedBinding.state c.request m surrounding
    ∃ t,exec (Typed.algebra rest (Native.ops p Bool.true)) (Kernels.cached (Typed.cLayout c.request)) s=some t ∧
      t.mem=(Typed.cGraphTarget p c.request s).mem ∧
      (∀ r o,Typed.COutside c.request s r o → t.mem r o=s.mem r o) ∧
      (∀ o,t.mem "K" o=s.mem "K" o) ∧
      (∀ i pi o,o<(c.tensor (TypedBinding.payloadRole i)).shape.prod →
        Physical.load m (.other c.request.device) (Checked.base c.request (TypedBinding.payloadRole i)+o)=
          TypedBinding.logical c objects m i pi o) ∧
      (∀ key odd i,surrounding.pids 0<c.request.qHeads → Typed.cActive (Typed.cLayout c.request) surrounding key i →
        Physical.load (Physical.native p c m) (.other c.request.device)
          (destination c.request (Physical.indices c.request m) (TypedBinding.cWork c.request surrounding key i) odd)=
          ((t.mem (Typed.cRegion (Typed.cLayout c.request) key)
            (Typed.cOffset (Typed.cLayout c.request) s key odd i)).read .real).value) ∧
      (∀ r address,(∀ w,c.request.Active w → ¬Writes c.request (Physical.indices c.request m) w r (address/4)) →
        Physical.native p c m r address=m r address) := by
  dsimp only
  obtain ⟨t,source,memory,_,frame,keys⟩ := rotary_emb_nopad_cached_fp32_equiv rest p c _ accepted cached
    (TypedBinding.state c.request m surrounding) (TypedBinding.metadata_matches _ _ _) token nonempty
  have same : t=(Typed.cachedExecution rest (Native.ops p Bool.true) (Typed.cLayout c.request)
      (TypedBinding.state c.request m surrounding)).val :=
    Option.some.inj (source.symm.trans (Typed.cachedExecution rest (Native.ops p Bool.true)
      (Typed.cLayout c.request) (TypedBinding.state c.request m surrounding)).property)
  refine ⟨t,source,memory,frame,keys,TypedBinding.direct c objects m accepted,?_,Physical.native_frame p c m accepted⟩
  subst t
  intro key odd i head live
  exact TypedBinding.c_physical rest p c m accepted cached surrounding token nonempty head key odd i live

#axiomsClean rotary_emb_nopad_checked_uncached_real
#axiomsClean rotary_emb_nopad_checked_cached_real
#axiomsClean rotary_emb_nopad_checked_uncached_fp32
#axiomsClean rotary_emb_nopad_checked_cached_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.TensorBridge
