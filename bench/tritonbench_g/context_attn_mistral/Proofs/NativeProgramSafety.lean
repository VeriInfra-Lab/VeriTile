import bench.tritonbench_g.context_attn_mistral.Proofs.NativeLoopSafety

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral Formula
open VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program (Regions)
open VeriTile.Bench.Repairs.ContextAttnMistral KernelReference
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem epilogue (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (initialState s : State α) (old : Acc p α)
    (live : Matches p r o (inputs p r initialState) old s)
    (bounds : RegionName → Nat) (access : Bounds p r (initialState.pids 0) (inputs p r initialState) bounds) :
    Safety.traceSafeList (engine o.algebra) bounds (parts p r .fp32 scale).after s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps:=500000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,FPBackend.Ops.algebra,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,numericLt,numericLe,natLt,natLe,
    bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,integralFloorDiv,integralMod,
    State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13]
  intro i d on
  exact access.output i d on

theorem loop (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α)
    (live : Matches p r o (inputs p r initialState) (initial p o) s) (memory : s.mem=initialState.mem)
    (bounds : RegionName → Nat) (access : Bounds p r (initialState.pids 0) (inputs p r initialState) bounds) :
    Safety.stepSafe (engine o.algebra) bounds
      (.forRangeInt "start_n" (parts p r .fp32 scale).start (parts p r .fp32 scale).stop
        (parts p r .fp32 scale).stride (parts p r .fp32 scale).body) s := by
  have inv : Safety.rangeIntSafe (engine o.algebra) bounds "start_n" 0
      (stop p (inputs p r initialState)) p.BN (parts p r .fp32 scale).body s := by
    apply Safety.rangeIntSafe_of_invariant (engine o.algebra) bounds "start_n"
      (stop p (inputs p r initialState)) p.BN (parts p r .fp32 scale).body
      (fun cur t=>0≤cur ∧ ∃ acc,Matches p r o (inputs p r initialState) acc t ∧ t.mem=initialState.mem)
    · intro cur on t ht
      obtain ⟨nonnegative,acc,matched,unchanged⟩:=ht
      exact body p r scale o hBN initialState t acc matched unchanged cur nonnegative on bounds access
    · intro cur on t ht u executed
      obtain ⟨nonnegative,acc,matched,unchanged⟩:=ht
      obtain ⟨v,ran,matched',unchanged'⟩:=KernelReference.body p r scale o hBN initialState t acc matched unchanged cur
      have same:u=v:=Option.some.inj (executed.symm.trans ran)
      subst u
      exact ⟨by omega,_,matched',unchanged'⟩
    · exact ⟨by omega,_,live,memory⟩
  rcases live with ⟨_,hb,_,_,_,_,_,_,_,_,_,_,_,hm⟩
  simp only [Safety.stepSafe_unfold]
  simp [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,Source.intRange?,ComputeKernel.surfaceBody,
    Safety.safeOp_unfold,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,
    resolvePrecision,numeric,bop,hb,hm,stop] at inv ⊢
  exact inv

theorem complete (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (s : State α) (bounds : RegionName → Nat)
    (access : Bounds p r (s.pids 0) (inputs p r s) bounds) :
    TraceSafe (engine o.algebra) bounds (program p r .fp32 scale) s := by
  change Safety.traceSafeList _ _ _ _
  rw [source_split,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude p r scale o s bounds access,?_⟩
  intro pre executed
  obtain ⟨t,ran,live,memory⟩:=KernelReference.prelude p r scale o s
  have same:pre=t:=Option.some.inj (executed.symm.trans ran)
  subst pre
  simp only [List.singleton_append,Safety.traceSafeList_cons]
  refine ⟨loop p r scale o hBN s t live memory bounds access,?_⟩
  intro post executed
  obtain ⟨u,ran,live',memory'⟩:=KernelReference.loop p r scale o hBN s t live memory
  have same:post=u:=Option.some.inj (executed.symm.trans ran)
  subst post
  exact epilogue p r scale o s u _ live' bounds access

#axiomsClean epilogue
#axiomsClean loop
#axiomsClean complete
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeSourceSafety
