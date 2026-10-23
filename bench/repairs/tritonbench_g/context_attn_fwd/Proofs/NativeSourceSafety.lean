import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeArrayBridge

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeSourceSafety
open VeriTile.Triton FP.Structural NativeFormula FPSuccess FPReference
noncomputable section
open Classical
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds over the independent arrays, including only masked reads reached
by the signed key loop. The proof below checks the actual source addresses. -/
structure Bounds (p : Layout) (r : Regions) (a : FPReference.Inputs α)
    (caps : RegionName → Nat) : Prop where
  start : a.batch < caps r.start
  sequence : a.batch < caps r.sequence
  prompt : a.batch < caps r.prompt
  query : ∀ off,off ∈ queryReads p a → off < caps r.query
  key : ∀ off,off ∈ keyReads p a → off < caps r.key
  value : ∀ off,off ∈ valueReads p a → off < caps r.value
  output : ∀ i,NativeFormula.active p a i → NativeFormula.address p a i < caps r.out

theorem prelude (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (s : State α) (caps : RegionName → Nat) (hb : Bounds p r (FPReference.inputs p r s) caps) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) caps (parts p r scale).before s := by
  have hq : ∀ i : TileIndex [p.BM,p.D],NativeFormula.active p (FPReference.inputs p r s) i →
      (((FPSuccess.start p r s+FPSuccess.row p s i)*p.qRow+
        (s.pids 1%p.heads*p.qHead : Nat)+(i.2.1.val*p.qDim : Nat)).toNat) < caps r.query := by
    intro i hi
    apply hb.query
    apply List.mem_filterMap.mpr
    refine ⟨i,TileShape.mem_allIndices _ i,?_⟩
    change (if NativeFormula.active p (FPReference.inputs p r s) i then _ else _) = _
    rw [if_pos hi]
    rfl
  simp (config := {maxSteps := 500000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,bop]
  refine ⟨hb.prompt,hb.start,hb.sequence,?_⟩
  intro i d hi
  simpa only [FPSuccess.start,FPSuccess.row,Nat.cast_mul,Nat.cast_add] using hq (i,d,()) hi

theorem body (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (positive : 0 < p.BN) (initial s : State α)
    (old : Acc p α) (live : Matches p M (FPReference.inputs p r initial) old s)
    (caps : RegionName → Nat) (hb : Bounds p r (FPReference.inputs p r initial) caps)
    (cur : Int) (reached : cur ∈ keySteps p 0
      ((blockMask p (FPReference.inputs p r initial) : Int)*keyEnd p (FPReference.inputs p r initial))) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) caps (parts p r scale).body
      (s.setReg "start_n" .int [] (fun _ => cur)) := by
  have hk : ∀ i : TileIndex [p.D,p.BN],cur+i.2.1.val < keyEnd p (FPReference.inputs p r initial) →
      (((((FPReference.inputs p r initial).batch*p.kBatch : Nat))+(cur+i.2.1.val)*p.kRow+
        ((FPReference.inputs p r initial).kvHead*p.kHead : Nat)+(i.1.val*p.kDim : Nat)).toNat) < caps r.key := by
    intro i hi
    apply hb.key
    apply List.mem_flatMap.mpr
    refine ⟨cur,reached,List.mem_filterMap.mpr ⟨i,TileShape.mem_allIndices _ i,?_⟩⟩
    simp [hi]
  have hv : ∀ i : TileIndex [p.BN,p.D],cur+i.1.val < keyEnd p (FPReference.inputs p r initial) →
      (((((FPReference.inputs p r initial).batch*p.vBatch : Nat))+(cur+i.1.val)*p.vRow+
        ((FPReference.inputs p r initial).kvHead*p.vHead : Nat)+(i.2.1.val*p.vDim : Nat)).toNat) < caps r.value := by
    intro i hi
    apply hb.value
    apply List.mem_flatMap.mpr
    refine ⟨cur,reached,List.mem_filterMap.mpr ⟨i,TileShape.mem_allIndices _ i,?_⟩⟩
    simp [hi]
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp (config := {maxSteps := 1000000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    dots.qk_bound,dots.pv_bound,TileShape.axisDim,positive,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro i j hi
    simpa only [Nat.cast_mul] using hk (i,j,()) hi
  · intro i j hi
    simpa only [Nat.cast_mul] using hv (i,j,()) hi

theorem epilogue (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (initial s : State α) (old : Acc p α) (live : Matches p M (FPReference.inputs p r initial) old s)
    (caps : RegionName → Nat) (hb : Bounds p r (FPReference.inputs p r initial) caps) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) caps (parts p r scale).after s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp (config := {maxSteps := 300000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    bop,Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro i d hi
    simpa only [NativeFormula.address,Nat.cast_mul] using hb.output (i,d,()) hi
  · split <;> trivial

end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeSourceSafety
