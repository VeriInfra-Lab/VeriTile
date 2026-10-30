import bench.tritonbench_g.chunk_retention.Sources.KernelPacket
import bench.tritonbench_g.chunk_retention.Proofs.Numeric

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelPacket
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelStaging
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000

/-- Independent scalar FMA targets, with every source cast and final scale in
its original position. There is no dot, block, or kernel-result callback. -/
def nativeTarget (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α)) : Observation a.shape→α
  | .history n i=>ops.cast (Reference.history ops a.shape (KernelNative.H.xs ops literal a.shape a.seed s) (s.pids 2) n
      (ForwardHSource.feature a.shape s i.1.val) (ForwardHSource.column a.shape s i.2.1.val))
  | .finalState i=>Reference.finalState ops a.shape (KernelNative.H.xs ops literal a.shape a.seed s) (s.pids 2)
      (ForwardHSource.feature a.shape s i.1.val) (ForwardHSource.column a.shape s i.2.1.val)
  | .output i=>Body.output ops a.shape (KernelNative.O.xs ops literal a.shape s) (literal a.scale)
      (s.pids 2) (s.pids 1) (KernelNative.O.history ops literal a.shape s)
      i.1.val (ForwardOSource.column a.shape s i.2.1.val)
  | .dhistory n i=>ops.cast (Reference.reverseCarry ops a.shape
      (KernelNative.BH.xs ops literal a.shape a.seed s) (literal a.scale) (s.pids 2) n
      (BackwardHSource.feature a.shape s i.1.val) (BackwardHSource.column a.shape s i.2.1.val))
  | .query i=>Body.dq ops a.shape (KernelNative.QKV.xs ops literal a.shape s) (literal a.scale)
      (s.pids 2) (s.pids 1) (KernelNative.QKV.history ops literal a.shape s "h")
      i.1.val (BackwardQKVSource.feature a.shape s i.2.1.val)
  | .key i=>Body.dk ops a.shape (KernelNative.QKV.xs ops literal a.shape s) (literal a.scale)
      (s.pids 2) (s.pids 1) (KernelNative.QKV.history ops literal a.shape s "dh")
      i.1.val (BackwardQKVSource.feature a.shape s i.2.1.val)
  | .valuePart f i=>Body.dvPartial ops a.shape (KernelNative.QKV.xs ops literal a.shape s) (literal a.scale)
      (s.pids 0) (s.pids 2) (s.pids 1) (KernelNative.QKV.history ops literal a.shape s "dh")
      i.1.val (BackwardQKVSource.column a.shape f i.2.1.val)
def NativeResult (ops : Ops α) (literal : ℝ→α) (a : Call) (s : State (Term α))
    (read : RegionName→Nat→α) : Prop :=
  ∀ o,Active a s o → read o.region (o.offset s)=nativeTarget ops literal a s o
def nativeDot (a : Call) : DotPrimitive (Term α) := match a.stage with
  | .backwardQKV=>gradientDot
  | _=>dot

theorem native_complete_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (a : Call) (s : State (Term α))
    (tag : a.stage=.backwardQKV → KernelNative.QKV.ReadTags a.shape s) (legal : Legal a s) :
    ∃ t,FP.Structural.exec (engine (withDot (algebra ops.zero) (nativeDot a))) a.kernel s=some t ∧
      NativeResult ops literal a s (fun r off=>lower ops literal ((t.mem r off).read .real)) ∧ Frame a s t := by
  cases hs : a.stage with
  | forwardH=>
    obtain ⟨t,ht,hh,hf,fr⟩:=KernelNative.H.source_result ops literal laws castZero a.shape a.seed a.final s
    refine ⟨t,by simpa only [Call.kernel,nativeDot,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      · exact hh _ _ ho.1 ho.2
      · exact hf ho.1 _ ho.2
    · simpa only [RawFrame,hs] using fr
  | forwardO=>
    obtain ⟨t,ht,hv,fr⟩:=KernelNative.O.source_result ops literal laws a.shape s a.scale
    refine ⟨t,by simpa only [Call.kernel,nativeDot,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      exact hv _ ho
    · simpa only [RawFrame,hs] using fr
  | backwardH=>
    obtain ⟨t,ht,hv,fr⟩:=KernelNative.BH.source_result ops literal laws castZero a.shape a.seed s a.scale
    refine ⟨t,by simpa only [Call.kernel,nativeDot,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      exact hv _ _ ho.1 ho.2
    · simpa only [RawFrame,hs] using fr
  | backwardQKV=>
    have hc : s.pids 1<a.shape.chunks := by
      have hlegal:=legal
      simp only [Legal,hs] at hlegal
      exact hlegal.2.2.2
    obtain ⟨t,ht,hq,hk,hv,fr⟩:=KernelNative.QKV.source_result_local ops literal laws a.shape s (tag hs) a.scale hc
    refine ⟨t,by simpa only [Call.kernel,nativeDot,hs] using ht,?_,frame_of_raw a s t ?_⟩
    · intro o ho
      cases o <;> simp only [Active,hs,reduceCtorEq,false_and,true_and] at ho
      · exact hq _ ho
      · exact hk _ ho
      · exact hv _ ho.1 _ ho.2
    · simpa only [RawFrame,hs] using fr

/-- The selected source and software IEEE observation graph meet through only
fixed-width scalar ports, literal conversions, and ordinary launch guards. -/
theorem fp_complete_local (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→BitVec 32)
    (literals : LiteralLaws (wordOps (Numeric.nativeISA ports)) literal)
    (a : Call) (s : State (Term (BitVec 32))) (tag : a.stage=.backwardQKV → KernelNative.QKV.ReadTags a.shape s) (legal : Legal a s) :
    ∃ t,FP.Structural.exec (engine (withDot (algebra (0 : BitVec 32)) (nativeDot a))) a.kernel s=some t ∧
      NativeResult (wordOps (Numeric.referenceISA u)) literal a s
        (fun r off=>lower (wordOps (Numeric.nativeISA ports)) literal ((t.mem r off).read .real)) ∧ Frame a s t := by
  obtain ⟨t,ht,hr,hf⟩:=native_complete_local (wordOps (Numeric.nativeISA ports)) literal literals rfl a s tag legal
  refine ⟨t,ht,?_,hf⟩
  simpa only [Numeric.isa_eq u ports laws] using hr

/-- Original global-provenance interface, unchanged. -/
theorem native_complete (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (a : Call) (s : State (Term α))
    (tag : KernelNative.QKV.Tagged s) (legal : Legal a s) :
    ∃ t,FP.Structural.exec (engine (withDot (algebra ops.zero) (nativeDot a))) a.kernel s=some t ∧
      NativeResult ops literal a s (fun r off=>lower ops literal ((t.mem r off).read .real)) ∧ Frame a s t :=
  native_complete_local ops literal laws castZero a s (fun _=>KernelNative.QKV.ReadTags.of_tagged a.shape s tag) legal

/-- Original global-provenance interface, unchanged. -/
theorem fp_complete (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→BitVec 32)
    (literals : LiteralLaws (wordOps (Numeric.nativeISA ports)) literal)
    (a : Call) (s : State (Term (BitVec 32))) (tag : KernelNative.QKV.Tagged s) (legal : Legal a s) :
    ∃ t,FP.Structural.exec (engine (withDot (algebra (0 : BitVec 32)) (nativeDot a))) a.kernel s=some t ∧
      NativeResult (wordOps (Numeric.referenceISA u)) literal a s
        (fun r off=>lower (wordOps (Numeric.nativeISA ports)) literal ((t.mem r off).read .real)) ∧ Frame a s t :=
  fp_complete_local u ports laws literal literals a s (fun _=>KernelNative.QKV.ReadTags.of_tagged a.shape s tag) legal

#axiomsClean native_complete_local
#axiomsClean fp_complete_local
#specNonCircular nativeTarget avoiding [run,FP.Structural.exec,ForwardHSource.update,BackwardQKVSource.queryValue,gradientDot]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelPacket
