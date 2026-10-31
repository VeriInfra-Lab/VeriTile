import bench.tritonbench_g.attn_fwd_triton.Proofs.AccessSafety
import bench.tritonbench_g.attn_fwd_triton.Proofs.CursorGeometry

/-! Every source access of a complete attention program follows the declared
input and output windows. The two range loops use their actual predecessor
states and the existing successful execution lemmas. -/
namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph Model ReadLocality
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Windows (p : Config) (bounds : RegionName → Nat) : Prop where
  input : ∀ key, InputAddress p key → key.2 < bounds key.1
  output : ∀ c : Coordinate p, logicalAddress p c < bounds p.Out

theorem initial_bounds (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (bounds : RegionName → Nat) (windows : Windows p bounds) :
    InitialBounds p BD idx.1.val idx.2.val bounds := by
  constructor
  · intro row dim active hd
    exact windows.input _ (query_input p BD idx (row,dim,PUnit.unit) active hd)
  · exact windows.input _ (query_scale_input p idx)

theorem iteration_bounds (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : ReadLocality.Memory) (n : Nat) (active : n*64<p.context)
    (bounds : RegionName → Nat) (windows : Windows p bounds) :
    IterationBounds p (fixed p BD memory idx.1.val idx.2.val) (n*64)
      (cursor (fixed p BD memory idx.1.val idx.2.val) n) bounds := by
  constructor
  · intro i row dim
    exact windows.input _ (key_input p BD idx memory n i
      (by change i.2.1.val<p.context-n*64 at row; omega) dim)
  · exact windows.input _ (key_scale_input p BD idx memory n active)
  · intro i row dim
    exact windows.input _ (value_input p BD idx memory n i
      (by change i.1.val<p.context-n*64 at row; omega) dim)

theorem output_bounds (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : ReadLocality.Memory) (bounds : RegionName → Nat) (windows : Windows p bounds) :
    OutputBounds p (fixed p BD memory idx.1.val idx.2.val) bounds := by
  intro i row dim
  have active : outputActive p BD idx.1.val i = Bool.true := by
    simpa [outputActive] using And.intro row dim
  exact windows.output (coordinate p BD idx i active)

theorem range_safe (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : ReadLocality.Memory) (stage : p.stage=3) (second : Bool)
    (lo hi n : Nat) (stop : hi≤p.context) (old : Carry BD) (s : State Expr)
    (hb : Bound (fixed p BD memory idx.1.val idx.2.val) old s)
    (hl : LoopBound second (cursor (fixed p BD memory idx.1.val idx.2.val) n) lo hi s)
    (bounds : RegionName → Nat) (windows : Windows p bounds) :
    Safety.rangeIntSafe (engine backend) bounds "start_n" ((n*64 : Nat) : Int) (hi : Int) 64 (body p BD second) s := by
  rw [Safety.rangeIntSafe_unfold]
  simp only [OfNat.ofNat_ne_zero,if_false]
  by_cases active : n*64<hi
  · simp only [show ((n*64 : Nat) : Int)<(hi : Int) by omega,if_true]
    refine ⟨iteration_safe p BD stage second _ old _ lo hi (n*64) s hb hl bounds
      (iteration_bounds p BD idx memory n (lt_of_lt_of_le active stop) bounds windows),?_⟩
    obtain ⟨t,ran,hb',hl',_⟩ := iteration p BD stage second _ old _ lo hi (n*64) s hb hl
    rw [ran]
    rw [advance_cursor] at hl'
    have next : n*64+64=(n+1)*64 := by omega
    rw [←Int.natCast_add,next]
    exact range_safe p BD idx memory stage second lo hi (n+1) stop _ t hb' hl' bounds windows
  · have hn : ¬(n : Int)*64<(hi : Int) := by omega
    simp [hn]
termination_by hi-n*64
decreasing_by omega

theorem loop_safe (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : ReadLocality.Memory) (stage : p.stage=3) (second : Bool)
    (old : Carry BD) (s : State Expr)
    (hb : Bound (fixed p BD memory idx.1.val idx.2.val) old s)
    (hl : LoopBound second
      (startCursor (fixed p BD memory idx.1.val idx.2.val) second)
      (lower (fixed p BD memory idx.1.val idx.2.val) second)
      (upper (fixed p BD memory idx.1.val idx.2.val) second) s)
    (bounds : RegionName → Nat) (windows : Windows p bounds) :
    Safety.traceSafeList (engine backend) bounds (loopCode p BD second) s := by
  have h := range_safe p BD idx memory stage second _ _
    (startChunk (fixed p BD memory idx.1.val idx.2.val) second)
    (stop_bounded p BD memory idx second) old s hb
    (by simpa only [start_cursor] using hl) bounds windows
  rw [←lower_chunk] at h
  rcases hl with ⟨_,_,_,hlo,hhi⟩
  unfold loopCode
  apply Safety.traceSafeList_cons_intro ?_ (fun _ _ => Safety.traceSafeList_nil _ _ _)
  cases second
  all_goals simp only [lowerName,upperName,lowerValue,lower,Bool.false_eq_true,Bool.true_eq,↓reduceIte] at hlo hhi
  all_goals simpa [stage,lowerOp,lowerName,lowerValue,upperName,Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,hlo,hhi,
    Option.some.injEq,true_and] using h

theorem source_safe (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (stage : p.stage=3) (initial : State Expr)
    (bounds : RegionName → Nat) (windows : Windows p bounds) :
    TraceSafe (engine backend) bounds (source p BD Bool.false) (atProgram p initial idx) := by
  let s := atProgram p initial idx
  let f := fixed p BD initial.mem idx.1.val idx.2.val
  have assembled : code p BD = (code p BD).take 27 ++
      setupCode p BD Bool.false ++ loopCode p BD Bool.false ++
      setupCode p BD Bool.true ++ loopCode p BD Bool.true ++ tail p BD Bool.false := by
    by_cases hd : p.stage=2
    all_goals simp [code,source,AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,
      AttnFwdTriton.ControlView.compute,setupCode,loopCode,body,lowerOp,lowerName,upperName,tail,hd]
  obtain ⟨s0,h0,b0,m0⟩ := initial_exact p BD s
  obtain ⟨s1,h1,b1,l1,m1⟩ := setup_exact p BD stage Bool.false f (initialCarry BD) s0 b0
  obtain ⟨s2,h2,b2,m2⟩ := loop_exact p BD stage Bool.false f (initialCarry BD) s1 b1 l1
  obtain ⟨s3,h3,b3,l3,m3⟩ := setup_exact p BD stage Bool.true f _ s2 b2
  unfold TraceSafe
  rw [←AttnFwdTriton.ControlView.compute_safe (engine backend) 32 6 _ (code_found p BD) bounds s]
  change Safety.traceSafeList (engine backend) bounds (code p BD) s
  rw [assembled]
  simp only [List.append_assoc,Safety.traceSafeList_append]
  refine ⟨initial_safe p BD s bounds (initial_bounds p BD idx bounds windows),?_⟩
  intro u hu
  have eu := Option.some.inj (hu.symm.trans h0)
  subst u
  refine ⟨setup_safe p BD stage Bool.false f _ s0 b0 bounds,?_⟩
  intro u hu
  have eu := Option.some.inj (hu.symm.trans h1)
  subst u
  refine ⟨loop_safe p BD idx initial.mem stage Bool.false _ s1 b1 l1 bounds windows,?_⟩
  intro u hu
  have eu := Option.some.inj (hu.symm.trans h2)
  subst u
  refine ⟨setup_safe p BD stage Bool.true f _ s2 b2 bounds,?_⟩
  intro u hu
  have eu := Option.some.inj (hu.symm.trans h3)
  subst u
  refine ⟨loop_safe p BD idx initial.mem stage Bool.true _ s3 b3 l3 bounds windows,?_⟩
  intro u hu
  obtain ⟨s4,h4,b4,_⟩ := loop_exact p BD stage Bool.true f _ s3 b3 l3
  have eu := Option.some.inj (hu.symm.trans h4)
  subst u
  exact tail_safe p BD f _ s4 b4 bounds (output_bounds p BD idx initial.mem bounds windows)

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.SourceSafety
