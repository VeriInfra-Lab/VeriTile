import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeSemantics

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open FPSuccess
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- A materialized floating value contains no unconsumed compiler dot payload. -/
def Materialized : Expr → Prop
  | .leaf .. | .literal .. => True
  | .cast _ _ _ x => Materialized x
  | .binary _ _ _ x y => Materialized x ∧ Materialized y
  | .pendingDot .. => False
  | .dotAcc _ _ _ _ _ _ _ _ _ _ a b acc _ =>
      (∀ i, Materialized (a i)) ∧ (∀ i, Materialized (b i)) ∧ Materialized acc

def Memory (s : State Expr) : Prop :=
  ∀ d rg off, Materialized (toFloat d ((s.mem rg off).read d.toTileDType))

def Accumulator (c : Host.Config) (s : State Expr) : Prop :=
  ∃ v : Values Expr .real [c.M,c.N], s.regs .real [c.M,c.N] "acc"=some v ∧
    ∀ i, Materialized (v i)

theorem memory_real (s : State Expr) (hm : Memory s) (rg : RegionName) (off : Nat) :
    Materialized ((s.mem rg off).read .real) := hm .real rg off

theorem setup_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (ready : Base s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine (backend rest c).algebra) (parts h c).before s=some t ∧
      Ready h c t ∧ t.mem=s.mem ∧ Fixed s t ∧ Accumulator c t := by
  rcases ready with ⟨h0,h1,h2,h3,h4⟩
  obtain ⟨v0,h0⟩ := Option.isSome_iff_exists.mp h0
  obtain ⟨v1,h1⟩ := Option.isSome_iff_exists.mp h1
  obtain ⟨v2,h2⟩ := Option.isSome_iff_exists.mp h2
  obtain ⟨v3,h3⟩ := Option.isSome_iff_exists.mp h3
  obtain ⟨v4,h4⟩ := Option.isSome_iff_exists.mp h4
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,Ready,Base,Has,Fixed,
      h0,h1,h2,h3,h4,backend,algebra,Accumulator,Materialized]

theorem body_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (ready : Ready h c s) (hm : Memory s) (acc : Accumulator c s)
    (index : Nat) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine (backend rest c).algebra) (parts h c).body
        (s.setReg "k" .nat [] (fun _ => index))=some t ∧
      Ready h c t ∧ t.mem=s.mem ∧ Fixed s t ∧ Accumulator c t := by
  rcases ready with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,_,hseq⟩
  obtain ⟨v0,h0⟩ := Option.isSome_iff_exists.mp h0
  obtain ⟨v1,h1⟩ := Option.isSome_iff_exists.mp h1
  obtain ⟨v2,h2⟩ := Option.isSome_iff_exists.mp h2
  obtain ⟨v3,h3⟩ := Option.isSome_iff_exists.mp h3
  obtain ⟨v4,h4⟩ := Option.isSome_iff_exists.mp h4
  obtain ⟨v5,h5⟩ := Option.isSome_iff_exists.mp h5
  obtain ⟨v6,h6⟩ := Option.isSome_iff_exists.mp h6
  obtain ⟨v7,h7⟩ := Option.isSome_iff_exists.mp h7
  obtain ⟨v8,h8⟩ := Option.isSome_iff_exists.mp h8
  obtain ⟨v9,h9⟩ := Option.isSome_iff_exists.mp h9
  obtain ⟨v10,h10⟩ := Option.isSome_iff_exists.mp h10
  obtain ⟨v11,h11,hv11⟩ := acc
  have good := memory_real s hm
  simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
    ha,hb,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,integralFloorDiv,integralMod,Ready,Base,Has,Fixed,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,backend,algebra,consume,Accumulator]
  refine ⟨hseq,?_⟩
  intro i j
  change (∀ j, Materialized _) ∧ (∀ j, Materialized _) ∧ Materialized (v11 (i,j,PUnit.unit))
  refine ⟨?_,?_,hv11 (i,j,PUnit.unit)⟩
  all_goals intro j; dsimp only [Materialized]; split <;> simp_all [Materialized]

#axiomsClean setup_materialized
#axiomsClean body_materialized
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph

#guard_msgs (drop info) in
#auditModuleAxioms
