import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeMaterialized

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open FPSuccess
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def CellMaterialized (v : Cell Expr) : Prop := ∀ d, Materialized (toFloat d (v.read d.toTileDType))

theorem cell_materialized (d : FloatDType) (v : Expr) (hv : Materialized v) :
    CellMaterialized (.mk d.toTileDType (ofFloat d v)) := by
  intro target
  cases d <;> cases target <;>
    simp [Cell.read,ofFloat,toFloat,defaultValue,Materialized,hv]

theorem write_materialized (s : State Expr) (hm : Memory s) (rg : RegionName) (off : Nat)
    (v : Cell Expr) (hv : CellMaterialized v) : Memory (s.write rg off v) := by
  intro d r n
  simp only [State.write]
  split
  · exact hv d
  · exact hm d r n

theorem scatter_materialized {ι : Type} (s : State Expr) (hm : Memory s)
    (rg : RegionName) (off : ι → Nat) (value : ι → Cell Expr)
    (active : ι → Prop) [DecidablePred active] (jobs : List ι)
    (hv : ∀ i, active i → CellMaterialized (value i)) :
    Memory (jobs.foldl (fun t i => if active i then t.write rg (off i) (value i) else t) s) := by
  induction jobs generalizing s with
  | nil => exact hm
  | cons i jobs ih =>
    simp only [List.foldl_cons]
    split
    · exact ih _ (write_materialized s hm rg (off i) (value i) (hv i ‹_›))
    · exact ih s hm

theorem iterations_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (cur stop : Nat) (s : State Expr) (ready : Ready h c s) (hm : Memory s)
    (acc : Accumulator c s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, range (engine (backend rest c).algebra) "k" cur stop 1 (parts h c).body s=some t ∧
      Ready h c t ∧ t.mem=s.mem ∧ Fixed s t ∧ Accumulator c t := by
  by_cases less : cur<stop
  · obtain ⟨mid,hmid,hr,hmem,hfixed,hacc⟩ := body_materialized rest h c s ready hm acc cur ha hb
    have hmidmem : Memory mid := by simpa only [Memory,hmem] using hm
    obtain ⟨t,ht,htr,htm,htfixed,htacc⟩ :=
      iterations_materialized rest h c (cur+1) stop mid hr hmidmem hacc ha hb
    refine ⟨t,?_,htr,htm.trans hmem,hfixed.trans htfixed,htacc⟩
    rw [range,if_neg (by decide),if_pos less,hmid]
    exact ht
  · exact ⟨s,by rw [range,if_neg (by decide),if_neg less],ready,rfl,Fixed.refl s,acc⟩
termination_by stop-cur

theorem loop_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (ready : Ready h c s) (hm : Memory s) (acc : Accumulator c s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, step (engine (backend rest c).algebra)
      (.forRangeDyn "k" (parts h c).start (parts h c).stop (parts h c).stride (parts h c).body) s=some t ∧
      Ready h c t ∧ t.mem=s.mem ∧ Fixed s t ∧ Accumulator c t := by
  obtain ⟨t,ht,hr,hm,hfixed,hacc⟩ :=
    iterations_materialized rest h c 0 (Host.cdiv h.shape.K c.K) s ready hm acc ha hb
  refine ⟨t,?_,hr,hm,hfixed,hacc⟩
  have hs : FP.Structural.evalOp (engine (backend rest c).algebra) none (parts h c).start s=some (fun _ => (0 : Nat)) := by
    simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
  have he : FP.Structural.evalOp (engine (backend rest c).algebra) none (parts h c).stop s=some (fun _ => Host.cdiv h.shape.K c.K) := by
    simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
  have hd : FP.Structural.evalOp (engine (backend rest c).algebra) none (parts h c).stride s=some (fun _ => (1 : Nat)) := by
    simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
  simpa only [step,hs,he,hd,Option.bind_some] using ht

theorem epilogue_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (initial s : State Expr) (ready : Ready h c s) (controls : Controls h c initial s)
    (hm : Memory s) (acc : Accumulator c s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine (backend rest c).algebra) (parts h c).after s=some t ∧ Memory t := by
  rcases ready with ⟨⟨_,_,_,_,_⟩,h5,h6,h7,h8,h9,h10,_,hseq⟩
  rcases controls with ⟨h0,h1,h2,h3,h4⟩
  obtain ⟨v11,h11,hv11⟩ := acc
  cases seq : h.hasSeq
  · simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,h0,h1,h2,h3,h4,h11,
      FP.Structural.store,backend,algebra]
    apply scatter_materialized
    · exact hm
    · intro i hi
      apply cell_materialized
      exact hv11 i
  · obtain ⟨vs,hs⟩ := Option.isSome_iff_exists.mp (hseq seq)
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,h0,h1,h2,h3,h4,h11,hs,
      FP.Structural.store,backend,algebra]
    apply scatter_materialized
    · exact hm
    · intro i hi
      apply cell_materialized
      dsimp only [Materialized]
      split_ifs <;> simp_all [Materialized]

#axiomsClean loop_materialized
#axiomsClean epilogue_materialized
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph

#guard_msgs (drop info) in
#auditModuleAxioms
