import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeVocabulary

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
open VeriTile.Triton FP.Structural FPSuccess
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem range_good (p : Layout) (r : Regions) (scale : ℝ) (plan : Plan) (hBN : 0<p.BN)
    (cur stop : Int) (s : State Expr) (hs : Ready p s) (hl : Live p s) (hm : GoodMemory s) :
    ∃ t,rangeInt ((backend plan).withDefaultPrecision .fp32) "start_n" cur stop p.BN
      (parts p r scale).body s=some t ∧ Ready p t ∧ Live p t ∧ GoodMemory t := by
  apply signed_range_invariant ((backend plan).withDefaultPrecision .fp32) "start_n"
    (parts p r scale).body (fun t => Ready p t ∧ Live p t ∧ GoodMemory t) p.BN hBN ?_ cur stop s ⟨hs,hl,hm⟩
  intro i t ht
  obtain ⟨u,hu,hready,_,_⟩ := body_success p r scale (backend plan) (dots.support (scalar plan) p) hBN t ht.1 i
  obtain ⟨v,hv,hlive,hmem⟩ := body_good p r scale plan hBN t ht.1 ht.2.1 ht.2.2 i
  have he : u=v := Option.some.inj (hu.symm.trans hv)
  subst v
  exact ⟨u,hu,hready,hlive,hmem⟩

theorem loop_good (p : Layout) (r : Regions) (scale : ℝ) (plan : Plan) (hBN : 0<p.BN)
    (s : State Expr) (hs : Ready p s) (hl : Live p s) (hm : GoodMemory s) :
    ∃ t,step ((backend plan).withDefaultPrecision .fp32)
      (.forRangeInt "start_n" (parts p r scale).start (parts p r scale).stop
        (parts p r scale).stride (parts p r scale).body) s=some t ∧
      Ready p t ∧ Live p t ∧ GoodMemory t := by
  have he := hs.2.2.1
  have hn := hs.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  rcases Option.isSome_iff_exists.mp he with ⟨e,he⟩
  rcases Option.isSome_iff_exists.mp hn with ⟨n,hn⟩
  obtain ⟨t,ht,hrt,hlt,hmt⟩ := range_good p r scale plan hBN 0
    ((n PUnit.unit : Int)*e PUnit.unit) s hs hl hm
  refine ⟨t,?_,hrt,hlt,hmt⟩
  have hstart : evalOp ((backend plan).withDefaultPrecision .fp32) none (parts p r scale).start s=
      some (fun _ => (0 : Int)) := rfl
  have hstride : evalOp ((backend plan).withDefaultPrecision .fp32) none (parts p r scale).stride s=
      some (fun _ => p.BN) := rfl
  have hstop : evalOp ((backend plan).withDefaultPrecision .fp32) none (parts p r scale).stop s=
      some (fun _ => (n PUnit.unit : Int)*e PUnit.unit) := by
    change evalOp _ none (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask"))
      (.ref .int [] "block_end_loc")) s=_
    simp only [evalOp_unfold,hn,he,Option.bind,numeric]
    rfl
  simpa only [step,hstart,hstride,hstop,Option.bind_some] using ht

theorem write_good (s : State Expr) (hm : GoodMemory s) (rg : RegionName) (off : Nat)
    (v : Expr) (hv : Good v) : GoodMemory (s.write rg off (.mk .real v)) := by
  intro r o
  simp only [State.write]
  split
  · simpa using hv
  · exact hm r o

theorem masked_good {ι : Type} (is : List ι) (rg : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (value : ι → Expr) (hv : ∀ i,Good (value i))
    (s : State Expr) (hm : GoodMemory s) :
    GoodMemory (is.foldl (fun t i => if active i then t.write rg (off i) (.mk .real (value i)) else t) s) := by
  induction is generalizing s with
  | nil => exact hm
  | cons i is ih =>
    simp only [List.foldl_cons]
    split
    · exact ih _ (write_good s hm rg (off i) (value i) (hv i))
    · exact ih s hm

theorem epilogue_good (p : Layout) (r : Regions) (scale : ℝ) (plan : Plan)
    (s : State Expr) (hs : Ready p s) (hl : Live p s) (hm : GoodMemory s) :
    ∃ t,run ((backend plan).withDefaultPrecision .fp32) (parts p r scale).after s=some t ∧ GoodMemory t := by
  rcases hs with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases Option.isSome_iff_exists.mp h4 with ⟨d,h4⟩
  rcases Option.isSome_iff_exists.mp h5 with ⟨m,h5⟩
  rcases Option.isSome_iff_exists.mp h9 with ⟨l,h9⟩
  rcases Option.isSome_iff_exists.mp h10 with ⟨a,h10⟩
  rcases Option.isSome_iff_exists.mp h11 with ⟨start,h11⟩
  rcases Option.isSome_iff_exists.mp h12 with ⟨head,h12⟩
  rcases Option.isSome_iff_exists.mp h13 with ⟨length,h13⟩
  have gl := hl.2.2.1 l h9
  have ga := hl.2.2.2 a h10
  simp (config := {maxSteps := 1000000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,resolvePrecision,
    numeric,numericLt,natLt,ofFloat,toFloat,Region.cast,h4,h5,h9,h10,h11,h12,h13,
    backend,FPSuccess.TotalDots.install,dots,scalar,cast,nativeDType,store]
  apply masked_good
  · intro i
    exact ⟨by simp [SupportedBinary],ga i,gl _⟩
  · exact hm

/-- Successful complete source execution constructs only supported graph nodes;
there is no unsupported-operation assumption in this theorem. -/
theorem program_good (p : Layout) (r : Regions) (scale : ℝ) (plan : Plan) (hBN : 0<p.BN)
    (s : State Expr) (hm : GoodMemory s) :
    ∃ t,exec ((backend plan).withDefaultPrecision .fp32) (kernel p r .fp32 scale) s=some t ∧ GoodMemory t := by
  obtain ⟨a,ha,hra,_,_⟩ := prelude_success p r scale (backend plan) s
  obtain ⟨a',ha',hla,hma⟩ := prelude_good p r scale plan s hm
  have he : a=a' := Option.some.inj (ha.symm.trans ha')
  subst a'
  obtain ⟨b,hb,hrb,hlb,hmb⟩ := loop_good p r scale plan hBN a hra hla hma
  obtain ⟨t,ht,hmt⟩ := epilogue_good p r scale plan b hrb hlb hmb
  refine ⟨t,?_,hmt⟩
  change run _ (kernel p r .fp32 scale).surfaceBody s=some t
  rw [source_split,run_append,run_append,ha]
  simp only [Option.bind_some,run,hb]
  exact ht

theorem grid_good (c : Host.Call) (plan : Plan) (order : List (GridIndex c.grid))
    (s : State Expr) (hm : GoodMemory s) :
    ∃ t,FPSuccess.runGrid c (backend plan) order s=some t ∧ GoodMemory t := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,hm⟩
  | cons idx rest ih =>
    obtain ⟨t,ht,hmt⟩ := program_good c.layout c.regions (Host.scale c) plan
      (by rw [c.blockN]; exact Host.bm_positive c) (FPSuccess.atIndex s idx) hm
    obtain ⟨u,hu,hmu⟩ := ih t hmt
    exact ⟨u,by simp only [FPSuccess.runGrid,ht,Option.bind_some]; exact hu,hmu⟩

#print axioms grid_good
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
