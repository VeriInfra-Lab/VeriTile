import bench.tritonbench_g.llama_ff_triton.IO
import bench.tritonbench_g.llama_ff_triton.Proofs.Store

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.SourceSafety
open VeriTile.Triton FP.Structural Safety Execution
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (p : Parameters) (initial : State α) (bounds : RegionName → Nat) : Prop where
  input : ∀ k (i : TileIndex [16,64]), activeK p k i.2.1.val →
    aOffset p initial k i < bounds "A"
  weight1 : ∀ k (i : TileIndex [64,16]), activeK p k i.1.val →
    wOffset p initial k i < bounds "W1"
  weight3 : ∀ k (i : TileIndex [64,16]), activeK p k i.1.val →
    wOffset p initial k i < bounds "W3"
  rms : ∀ k (i : Fin 64), activeK p k i.val → k*64+i.val < bounds "RMS"
  output : ∀ i : TileIndex [16,16], active p initial i → outputOffset p initial i < bounds "OUT"

theorem prelude_safe (M : Algebra α) (p : Parameters) (s : State α) (bounds : RegionName → Nat) :
    traceSafeList M bounds (prelude p) s := by
  rw [prelude_half]
  simp [prelude,source,Source.kernel,Source.half,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
  all_goals repeat' first | exact True.intro | constructor | split

theorem body_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State α) (k : Nat) (h : Invariant a native p initial k current)
    (bytes : ByteValues initial) (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    traceSafeList a.model bounds (loopBody native p)
      (current.setReg "step" .int [] (fun _ => (k : Int))) := by
  have valid (weight : RegionName) (hw : weight ∈ (["RMS","W1","W3"] : List RegionName))
      (offset : Nat) (test : Prop) [Decidable test] :
      Float8E5Bits.isInt8 (if test then (initial.mem weight offset).read .int else 0)=Bool.true := by
    split_ifs
    · exact bytes weight hw offset
    · decide
  cases native <;> cases hp : p.byte
  all_goals
    simp [loopBody,compiledBody,dot1Index,dot3Index,carryNode,directWeightNode,
      body,source,Source.kernel,Source.half,Source.byte,hp,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,
      Option.bind,integralFloorDiv,integralMod,Arithmetic.model,
      h.memory,h.k,h.sak,h.w1k,h.w3k,h.rms,h.offk,
      h.aptrs,h.bptrs,h.cptrs,h.rptrs,h.acc1,h.acc2,h.sum,
      Broadcast.leftIndex,Broadcast.rightIndex,mask_eq,
      Int32Bits.valuesValid,Int32Bits.inRange,
      valid "RMS" (by simp),valid "W1" (by simp),valid "W3" (by simp)]
  all_goals
    refine ⟨?_,?_,?_,?_⟩
    · intro i j u hact; exact hb.input k (i,j,()) hact
    · intro j hact; exact hb.rms k j hact
    · intro i j u hact; exact hb.weight1 k (i,j,()) hact
    · intro i j u hact; exact hb.weight3 k (i,j,()) hact

theorem norm_safe (a : Arithmetic α) (p : Parameters) (s : State α) (bounds : RegionName → Nat) :
    traceSafeList a.model bounds ((epilogue p).take 2) s := by
  cases hp : p.byte <;>
    simp [epilogue,source,Source.kernel,Source.half,Source.byte,hp,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
  all_goals repeat' first | exact True.intro | constructor | split

theorem store_safe (a : Arithmetic α) (p : Parameters) (initial current : State α)
    (values : TileIndex [16,16] → α) (h : StoreContext p initial current)
    (hv : current.regs .real [16,16] "accumulator"=some values)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    traceSafeList a.model bounds ((epilogue p).drop 5) current := by
  have hep : (epilogue p).drop 5=(epilogue {p with byte:=Bool.false}).drop 5 := by
    rcases p with ⟨M,N,K,b,e⟩; cases b <;> rfl
  rw [hep]
  simp [epilogue,source,Source.kernel,Source.half,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    h.pidm,h.pidn,h.memory,h.m,h.n,h.outm,h.outn,hv,Arithmetic.model,
    Broadcast.leftIndex,Broadcast.rightIndex]
  intro i j hr hc
  apply hb.output (i,j,())
  constructor
  · change rowBlock p initial*16+i.val<p.rows
    apply Int.ofNat_lt.mp
    simpa only [Int.natCast_add,Int.natCast_mul,Nat.cast_ofNat] using hr
  · change columnBlock p initial*16+j.val<p.columns
    apply Int.ofNat_lt.mp
    simpa only [Int.natCast_add,Int.natCast_mul,Nat.cast_ofNat] using hc

theorem epilogue_safe (a : Arithmetic α) (p : Parameters) (initial current : State α)
    (h : Invariant a Bool.false p initial (steps p) current)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    traceSafeList a.model bounds (epilogue p) current := by
  have hep : epilogue p=epilogue {p with byte:=Bool.false} := by
    rcases p with ⟨M,N,K,b,e⟩; cases b <;> rfl
  rw [hep]
  simp [epilogue,source,Source.kernel,Source.half,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    h.pidm,h.pidn,h.acc1,h.acc2,h.sum,h.memory,h.m,h.n,h.k,h.outm,h.outn,Arithmetic.model,
    Broadcast.leftIndex,Broadcast.rightIndex]
  intro i j hr hc
  apply hb.output (i,j,())
  constructor
  · change rowBlock p initial*16+i.val<p.rows
    apply Int.ofNat_lt.mp
    simpa only [Int.natCast_add,Int.natCast_mul,Nat.cast_ofNat] using hr
  · change columnBlock p initial*16+j.val<p.columns
    apply Int.ofNat_lt.mp
    simpa only [Int.natCast_add,Int.natCast_mul,Nat.cast_ofNat] using hc

end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.SourceSafety
