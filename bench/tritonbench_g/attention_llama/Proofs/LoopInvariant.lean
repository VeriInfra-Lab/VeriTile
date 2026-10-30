import bench.tritonbench_g.attention_llama.Proofs.TileStep
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.Repairs.AttentionLlama.LoopInvariant
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open SourceExecution MaskCoverage
open scoped BigOperators
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

structure Context where
  initial : BlockState
  Q : RegionName
  K : RegionName
  V : RegionName
  scale : ℝ
  sqz : Nat
  sqh : Nat
  sqm : Nat
  sqk : Nat
  skz : Nat
  skh : Nat
  skn : Nat
  skk : Nat
  svz : Nat
  svh : Nat
  svk : Nat
  svn : Nat
  NH : Nat
  H : Nat
  N : Nat
  SP : Nat
  D : Nat
  causal : Bool

namespace Context
noncomputable def query (c : Context) : TileIndex [64,c.D] → ℝ :=
  alQTileG c.initial c.Q c.NH c.sqz c.sqh c.sqm c.sqk c.H 64 c.D
noncomputable def key (c : Context) (k : Nat) (d : Fin c.D) : ℝ :=
  c.initial.readMem c.K (alBase c.initial c.NH c.skz c.skh + k*c.skn + d.val*c.skk)
noncomputable def value (c : Context) (k : Nat) (d : Fin c.D) : ℝ :=
  c.initial.readMem c.V (alBase c.initial c.NH c.svz c.svh + k*c.svk + d.val*c.svn)
def row (c : Context) (r : Fin 64) : Nat := c.initial.pids 0 * 64 + r.val
noncomputable def score (c : Context) (r : Fin 64) (k : Nat) : ℝ :=
  (∑ d : Fin c.D, c.query (r,d,PUnit.unit) * c.key k d) * c.scale
def keys (c : Context) (r : Fin 64) : Nat → Finset Nat :=
  blockKeys c.N c.causal c.SP (c.row r)
noncomputable def keyTile (c : Context) (n : Nat) : TileIndex [64,c.D] → ℝ :=
  fun idx => if n*64+idx.1.val < c.N then c.key (n*64+idx.1.val) idx.2.1 else 0
noncomputable def valueTile (c : Context) (n : Nat) : TileIndex [64,c.D] → ℝ :=
  fun idx => if n*64+idx.1.val < c.N then c.value (n*64+idx.1.val) idx.2.1 else 0
noncomputable def scores (c : Context) (n : Nat) : Tile .real [64,64] :=
  ⟨fun idx => if n*64+idx.2.1.val < c.N ∧ visible c.causal c.SP (c.row idx.1) (n*64+idx.2.1.val)
    then some (c.score idx.1 (n*64+idx.2.1.val)) else none⟩
end Context

noncomputable def maximumTile (n : Nat) (m : Fin 64 → ℝ) : Tile .real [64] :=
  ⟨fun idx => if n = 0 then none else some (m idx.1)⟩

structure Invariant (c : Context) (n : Nat) (st : BlockState) where
  m : Fin 64 → ℝ
  l : Fin 64 → ℝ
  a : TileIndex [64,c.D] → ℝ
  pids : st.pids = c.initial.pids
  mem : st.mem = c.initial.mem
  undef : ∀ rg o, st.undef rg o = 0
  maximum : st.regs .real [64] "m_prev" = some (maximumTile n m)
  denominator : st.regs .real [64] "l_prev" = some ⟨fun idx => some (l idx.1)⟩
  accumulator : st.regs .real [64,c.D] "acc" = some ⟨fun idx => some (a idx)⟩
  query : st.regs .real [64,c.D] "q" = some ⟨fun idx => some (c.query idx)⟩
  rows : st.regs .nat [64] "offs_m" = some (Tile.vec c.row)
  columns : st.regs .nat [64] "offs_n" = some (Tile.vec (fun j : Fin 64 => j.val))
  batch : st.regs .nat [] "batch_id" = some (Tile.scalar (c.initial.pids 1 / c.NH))
  head : st.regs .nat [] "off_hz" = some (Tile.scalar (c.initial.pids 1 % c.NH))
  keyPtr : st.regs .ptr [64,c.D] "k_ptrs" =
    some (alPtrTile c.K (alBase c.initial c.NH c.skz c.skh) c.skn c.skk (n*64) 64 c.D)
  valuePtr : st.regs .ptr [64,c.D] "v_ptrs" =
    some (alPtrTile c.V (alBase c.initial c.NH c.svz c.svh) c.svk c.svn (n*64) 64 c.D)
  numerical : ∀ r d, TailMath.Consistent (c.score r) (fun k => c.value k d)
    (TailMath.processed (c.keys r) n) ⟨m r,l r,a (r,d,PUnit.unit)⟩

theorem scaled_dot {D : Nat} (q k : TileIndex [64,D] → ℝ) (sc : ℝ)
    (r j : Fin 64) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        (⟨fun _ : TileIndex [64,64] => some (0 : ℝ)⟩ : Tile .real [64,64])
        (Tile.dot [] (⟨fun idx => some (q idx)⟩ : Tile .real [64,D])
          (Tile.transpose [] (⟨fun idx => some (k idx)⟩ : Tile .real [64,D]))))
      (Tile.scalar (some sc : WithBot ℝ))).data (r,j,PUnit.unit) =
        some ((∑ d : Fin D, q (r,d,PUnit.unit) * k (j,d,PUnit.unit)) * sc) := by
  have h := al_qk0_cell 64 64 D 1 q k sc 0 (by decide) r j
  simpa [StreamingAccumulator.blockIndex, StreamingAccumulator.scaledScore, mul_comm] using h

theorem score_cell (c : Context) (n : Nat) (r j : Fin 64) (hj : n*64+j.val < c.N) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
        (⟨fun _ : TileIndex [64,64] => some (0 : ℝ)⟩ : Tile .real [64,64])
        (Tile.dot [] (⟨fun idx => some (c.query idx)⟩ : Tile .real [64,c.D])
          (Tile.transpose [] (⟨fun idx => some (c.keyTile n idx)⟩ : Tile .real [64,c.D]))))
      (Tile.scalar (some c.scale : WithBot ℝ))).data (r,j,PUnit.unit) =
        some (c.score r (n*64+j.val)) := by
  rw [scaled_dot]
  simp only [Context.keyTile, hj, ↓reduceIte, Context.score]

theorem prior_valid (c : Context) (n : Nat) (st : BlockState) (hD : 0 < c.D)
    (h : Invariant c n st) (r : Fin 64) :
    (maximumTile n h.m).data (r,PUnit.unit) = some (h.m r) ∨ h.l r = 0 := by
  by_cases hn : n = 0
  · right
    have hd := (h.numerical r ⟨0,hD⟩).1
    simpa [hn, TailMath.processed, TailMath.mass] using hd
  · left; simp [maximumTile, hn]

theorem selected_finite (c : Context) (n : Nat) (st : BlockState)
    (h : Invariant c n st) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64)
    (rm : Tile .real [64])
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64,64].length) (c.scores n) = some rm) :
    ∃ μ : Fin 64 → ℝ, ∀ r,
      (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rm
        (maximumTile n h.m)) rm (maximumTile n h.m)).data (r,PUnit.unit) = some (μ r) := by
  apply TileMath.finite_max _ _ _ hrm
  intro r
  by_cases hn : n = 0
  · right
    refine ⟨⟨0,by decide⟩, c.score r 0, ?_⟩
    have hv : visible c.causal c.SP (c.row r) 0 := by
      cases hc : c.causal
      · exact Or.inl rfl
      · right
        have hs := hSP hc
        simp only [Context.row]
        omega
    simp [Context.scores, hn, hv, hN]
  · left; exact ⟨h.m r, by simp [maximumTile, hn]⟩

theorem iteration (c : Context) (n : Nat) (st : BlockState)
    (h : Invariant c n st) (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64) :
    ∃ sf, stepStmts (sourceLoop c.scale c.skn c.svk c.N c.SP 64 64 c.D c.causal)
      (st.setReg "start_n" .nat [] (Tile.scalar (n*64))) = some sf ∧
      Nonempty (Invariant c (n+1) sf) := by
  let se := st.setReg "start_n" .nat [] (Tile.scalar (n*64))
  obtain ⟨sp, hpre, hpp, hpm, hpu, hpqk, hpbn, hpmp, hplp, hpacc,
      hpq, hpom, hpon, hpbi, hpoh, hpkp, hpvp⟩ :=
    prefix_steps_with_masked_keys c.scale c.N 64 64 c.D (n*64) se c.K
      (alBase c.initial c.NH c.skz c.skh) c.skn c.skk
      (⟨fun idx => some (c.query idx)⟩) (c.keyTile n)
      (by simp [se]) (by simpa [se] using h.columns) (by simpa [se] using h.query)
      (by simpa [se] using h.keyPtr)
      (by intro idx; simp only [Context.keyTile, Context.key, se, BlockState.setReg_readMem]
          unfold BlockState.readMem; rw [h.mem])
      (by simpa [se] using h.undef)
  let qk0 : Tile .real [64,64] := ⟨fun idx => if n*64+idx.2.1.val < c.N
    then some (c.score idx.1 (n*64+idx.2.1.val)) else none⟩
  have hqk0 : sp.regs .real [64,64] "qk" = some qk0 := by
    rw [hpqk]
    apply congrArg some
    ext idx
    rcases idx with ⟨r,j,u⟩; cases u
    by_cases hj : n*64+j.val < c.N
    · simp only [qk0,hj,↓reduceIte]
      exact score_cell c n r j hj
    · simp only [qk0,hj,↓reduceIte]; rfl
  let qk1 : Tile .real [64,64] := ⟨fun idx =>
    if n*64+idx.2.1.val+c.SP ≤ c.row idx.1 then qk0.data idx else none⟩
  let sw := if c.causal then sp.setReg "qk" .real [64,64] qk1 else sp
  have hwreg : ∀ (ty : TileDType) sh name, name ≠ "qk" →
      sw.regs ty sh name = sp.regs ty sh name := by
    intro ty sh name hn
    cases hc : c.causal <;> simp [sw,hc,BlockState.setReg_ne_name,hn,Ne.symm hn]
  have hwm : sw.mem = c.initial.mem := by
    cases hc : c.causal <;> simp only [sw,hc,Bool.false_eq_true,↓reduceIte]
    all_goals exact hpm.trans h.mem
  have hwp : sw.pids = c.initial.pids := by
    cases hc : c.causal <;> simp [sw,hc,hpp,se,h.pids]
  have hwu : ∀ rg o, sw.undef rg o = 0 := by
    cases hc : c.causal <;> simpa [sw,hc] using hpu
  have hwqk : sw.regs .real [64,64] "qk" = some (c.scores n) := by
    cases hc : c.causal
    · simp only [sw,hc,Bool.false_eq_true,↓reduceIte,hqk0]
      apply congrArg some
      ext idx
      simp [qk0,Context.scores,visible,hc]
    · simp only [sw,hc,↓reduceIte,BlockState.setReg_same]
      apply congrArg some
      ext idx
      simp only [qk1,qk0,Context.scores,visible,hc,Bool.true_eq_false,false_or]
      by_cases hn : n*64+idx.2.1.val < c.N <;>
        by_cases hv : n*64+idx.2.1.val+c.SP ≤ c.row idx.1 <;> simp [hn,hv]
  have hwbn : sw.regs .nat [64] "block_n_offs" =
      some (Tile.vec (fun j : Fin 64 => n*64+j.val)) := by
    rw [hwreg _ _ _ (by decide),hpbn]
  have hwmp : sw.regs .real [64] "m_prev" = some (maximumTile n h.m) := by
    rw [hwreg _ _ _ (by decide),hpmp]
    simpa [se] using h.maximum
  have hwlp : sw.regs .real [64] "l_prev" = some ⟨fun idx => some (h.l idx.1)⟩ := by
    rw [hwreg _ _ _ (by decide),hplp]
    simpa [se] using h.denominator
  have hwacc : sw.regs .real [64,c.D] "acc" = some ⟨fun idx => some (h.a idx)⟩ := by
    rw [hwreg _ _ _ (by decide),hpacc]
    simpa [se] using h.accumulator
  have hwkp : sw.regs .ptr [64,c.D] "k_ptrs" =
      some (alPtrTile c.K (alBase c.initial c.NH c.skz c.skh) c.skn c.skk (n*64) 64 c.D) := by
    rw [hwreg _ _ _ (by decide),hpkp]
    simpa [se] using h.keyPtr
  have hwvp : sw.regs .ptr [64,c.D] "v_ptrs" =
      some (alPtrTile c.V (alBase c.initial c.NH c.svz c.svh) c.svk c.svn (n*64) 64 c.D) := by
    rw [hwreg _ _ _ (by decide),hpvp]
    simpa [se] using h.valuePtr
  obtain ⟨rm,hrm,sf,htail,hfp,hfm,hfu,mc,lc,lr,p,ac,hmcd,hpd,hlcd,hlrd,hacd,
      hmf,hlf,haf,hkf,hvf,hqf,homf,honf,hbif,hohf⟩ :=
    tail_steps_with_masked_values c.skn c.svk c.N 64 64 c.D (n*64) sw
      (by decide) (c.scores n) (maximumTile n h.m)
      (⟨fun idx => some (h.l idx.1)⟩) (⟨fun idx => some (h.a idx)⟩)
      c.K c.V (alBase c.initial c.NH c.skz c.skh) (alBase c.initial c.NH c.svz c.svh)
      c.skk c.svn (c.valueTile n) hwqk hwbn hwmp hwlp hwacc hwkp hwvp
      (by intro idx; simp only [Context.valueTile,Context.value]
          unfold BlockState.readMem; rw [hwm]) hwu
  have hrun : stepStmts (sourceLoop c.scale c.skn c.svk c.N c.SP 64 64 c.D c.causal) se = some sf := by
    unfold sourceLoop
    rw [stepStmts.append_some hpre,ProgramRewrite.constant_branch]
    cases hc : c.causal
    · simp only [Bool.false_eq_true,↓reduceIte,List.nil_append]
      rw [sourceTail_step]
      simpa [sw,hc] using htail
    · simp only [↓reduceIte,List.cons_append,List.nil_append]
      have he := al_where_causal_eval sp 64 64 (n*64) c.SP c.row qk0
        (by rw [hpom]; simpa [se] using h.rows) hpbn hqk0
      unfold alCausalWhereStmt
      rw [stepStmts.cons_some (stepStmt_assign_eq_some he),sourceTail_step]
      simpa [sw,hc,qk1] using htail
  obtain ⟨μ,hμ⟩ := selected_finite c n st h hN hSP rm hrm
  have hmc : ∀ r, mc.data (r,PUnit.unit) = some (μ r) := by
    intro r; rw [hmcd]; exact hμ r
  let next := fun (r : Fin 64) (d : Fin c.D) =>
    TailMath.step (c.score r) (fun k => c.value k d) (c.keys r n) (μ r)
      ⟨h.m r,h.l r,h.a (r,d,PUnit.unit)⟩
  let newL := fun r => (next r ⟨0,hD⟩).denominator
  let newA := fun idx : TileIndex [64,c.D] => (next idx.1 idx.2.1).accumulator
  have hlcell : ∀ r, lc.data (r,PUnit.unit) = some (newL r) := by
    intro r
    rw [hlcd]
    exact TileStep.denominator_cell (c.scores n) (maximumTile n h.m)
      (⟨fun idx => some (h.l idx.1)⟩) mc μ hmc c.N c.causal c.SP (c.row r) n
      (c.score r) r (h.m r) (h.l r) (h.a (r,⟨0,hD⟩,PUnit.unit))
      (fun _ => rfl) (prior_valid c n st hD h r) rfl
  have hacell : ∀ idx : TileIndex [64,c.D],
      (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame ac
        (Tile.dot [] p (⟨fun idx => some (c.valueTile n idx)⟩ : Tile .real [64,c.D]))).data idx =
        some (newA idx) := by
    rintro ⟨r,d,u⟩; cases u
    rw [hacd,hpd,hlrd,hlcd]
    convert TileStep.accumulator_cell (c.scores n) (maximumTile n h.m)
      (⟨fun idx => some (h.l idx.1)⟩) mc (⟨fun idx => some (h.a idx)⟩)
      (⟨fun idx => some (c.valueTile n idx)⟩) μ hmc c.N c.causal c.SP (c.row r) n
      (c.score r) (fun k => c.value k d) r d (h.m r) (h.l r) (h.a (r,d,PUnit.unit))
      (fun _ => rfl) (prior_valid c n st hD h r) rfl rfl (c.valueTile n) (fun _ => rfl) (fun _ => rfl) using 1 <;>
      norm_num [TileStep.accumulator,TileStep.rescaled,TileStep.prior,TileStep.normalized,
        TileStep.probability,TileStep.reciprocal,TileStep.denominator,newA,next,Context.keys]
  refine ⟨sf,hrun,⟨{
    m := μ, l := newL, a := newA
    pids := hfp.trans hwp
    mem := hfm.trans hwm
    undef := hfu
    maximum := ?_
    denominator := ?_
    accumulator := ?_
    query := ?_
    rows := ?_
    columns := ?_
    batch := ?_
    head := ?_
    keyPtr := ?_
    valuePtr := ?_
    numerical := ?_
  }⟩⟩
  · rw [hmf]; apply congrArg some; ext idx
    simpa [maximumTile] using hmc idx.1
  · rw [hlf]; apply congrArg some; ext idx; exact hlcell idx.1
  · rw [haf]; apply congrArg some; ext idx; exact hacell idx
  · rw [hqf,hwreg _ _ _ (by decide),hpq]; simpa [se] using h.query
  · rw [homf,hwreg _ _ _ (by decide),hpom]; simpa [se] using h.rows
  · rw [honf,hwreg _ _ _ (by decide),hpon]; simpa [se] using h.columns
  · rw [hbif,hwreg _ _ _ (by decide),hpbi]; simpa [se] using h.batch
  · rw [hohf,hwreg _ _ _ (by decide),hpoh]; simpa [se] using h.head
  · simpa [Nat.add_mul] using hkf
  · simpa [Nat.add_mul] using hvf
  · intro r d
    exact TailMath.step_consistent (c.score r) (fun k => c.value k d)
      (TailMath.processed (c.keys r) n) (c.keys r n) (μ r)
      ⟨h.m r,h.l r,h.a (r,d,PUnit.unit)⟩
      (blocks_disjoint c.N c.causal c.SP (c.row r) n) (h.numerical r d)

end VeriTile.Bench.Repairs.AttentionLlama.LoopInvariant
