import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeGeometry

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native.SourceSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph FPSuccess SourcePointers
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Active bounds only. Each field is discharged from the public IO windows;
the record is a proof helper, not a numerical or public domain assumption. -/
structure Bounds (h : Host.Call) (c : Host.Config) (d : Coordinates) (bounds : RegionName → Nat) : Prop where
  a : ∀ q,q<Host.cdiv h.shape.K c.K → ∀ i : Fin c.M, ∀ e : Fin c.K,
    SourceAccess.aMask h c d q i e=Bool.true → aOffset h c d q i e<bounds h.A
  b : ∀ q,q<Host.cdiv h.shape.K c.K → ∀ e : Fin c.K, ∀ j : Fin c.N,
    SourceAccess.bMask h c d q e j=Bool.true → bOffset h c d q e j<bounds h.B
  rowSeq : h.hasSeq=Bool.true → ∀ i : Fin c.M, row c d i<limit h d → seqOffset h d (row c d i)<bounds (Region.cast h.Seq)
  colSeq : h.hasSeq=Bool.true → ∀ j : Fin c.N, col c d j<limit h d → seqOffset h d (col c d j)<bounds (Region.cast h.Seq)
  out : ∀ i : Fin c.M, ∀ j : Fin c.N, SourceAccess.outMask h c d i j=Bool.true → outOffset h c d i j<bounds h.Out

theorem prologue {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (bounds : RegionName → Nat) (s : State α) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    Safety.traceSafeList (engine o.algebra) bounds (branch h c).before s := by
  cases causal : h.causal
  all_goals
    simp (config := {maxSteps := 300000}) [branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,causal,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeAlg,Safety.safeOp.eq_def,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod]

theorem setup {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (bounds : RegionName → Nat) (s : State α) (hc : SourcePointers.Controls d s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    Safety.traceSafeList (engine o.algebra) bounds (parts h c).before s := by
  rcases hc with ⟨h0,h1,h2,h3,h4⟩
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeAlg,Safety.safeOp.eq_def,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,h0,h1,h2,h3,h4]

theorem body {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (q : Nat) (bounds : RegionName → Nat) (s : State α) (hs : StateAt h c d q s)
    (aBound : ∀ i : Fin c.M, ∀ e : Fin c.K,SourceAccess.aMask h c d q i e=Bool.true → aOffset h c d q i e<bounds h.A)
    (bBound : ∀ e : Fin c.K, ∀ j : Fin c.N,SourceAccess.bMask h c d q e j=Bool.true → bOffset h c d q e j<bounds h.B)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    Safety.traceSafeList (engine o.algebra) bounds (parts h c).body (s.setReg "k" .nat [] (fun _=>q)) := by
  rcases hs with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  obtain ⟨acc,h11⟩ := Option.isSome_iff_exists.mp h11
  simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
    ha,hb,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeAlg,Safety.safeOp.eq_def,
    Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,integralFloorDiv,integralMod,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,SourceAccess.aMask,SourceAccess.bMask]
  exact ⟨fun i e hr hk=>aBound i e (by simpa [SourceAccess.aMask] using And.intro hr hk),
    fun e j hk hj=>bBound e j (by simpa [SourceAccess.bMask] using And.intro hk hj)⟩

theorem epilogue {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (q : Nat) (bounds : RegionName → Nat) (s : State α) (hs : StateAt h c d q s)
    (rowBound : h.hasSeq=Bool.true → ∀ i : Fin c.M, row c d i<limit h d → seqOffset h d (row c d i)<bounds (Region.cast h.Seq))
    (colBound : h.hasSeq=Bool.true → ∀ j : Fin c.N, col c d j<limit h d → seqOffset h d (col c d j)<bounds (Region.cast h.Seq))
    (outBound : ∀ i : Fin c.M, ∀ j : Fin c.N,SourceAccess.outMask h c d i j=Bool.true → outOffset h c d i j<bounds h.Out)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    Safety.traceSafeList (engine o.algebra) bounds (parts h c).after s := by
  rcases hs with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  obtain ⟨acc,h11⟩ := Option.isSome_iff_exists.mp h11
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeAlg,Safety.safeOp.eq_def,
      Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,hseq,FP.Structural.store]
  · intro i j hr hc
    have bound := outBound i j (by simpa [SourceAccess.outMask] using And.intro hr hc)
    convert bound using 1 <;> simp only [outOffset,SourcePointers.row,col,Region.cast] <;> cases h.shape.grouped <;> simp <;> congr 1 <;> ring
  · refine ⟨⟨?_,?_⟩,?_⟩
    · intro i hmask
      have bound := rowBound seq i (by simpa only [SourcePointers.row,limit,minimum] using hmask)
      convert bound using 1 <;> simp only [seqOffset,SourcePointers.row,Region.cast] <;> congr 1 <;> ring
    · intro j hmask
      have bound := colBound seq j (by simpa only [col,limit,minimum] using hmask)
      convert bound using 1 <;> simp only [seqOffset,col,Region.cast] <;> congr 1 <;> ring
    · intro i j hr hc
      have bound := outBound i j (by simpa [SourceAccess.outMask] using And.intro hr hc)
      convert bound using 1 <;> simp only [outOffset,SourcePointers.row,col,Region.cast] <;> cases h.shape.grouped <;> simp <;> congr 1 <;> ring

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native.SourceSafety
