import bench.tritonbench_g.iv_dependent_matmul.Proofs.Host

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.RealTarget
open VeriTile.Triton FP.Structural FPValue Kernels Whole
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

/-- Independent mathematical target, with no program registers or execution. -/
def mathematical (inner : Nat) (A B : Nat → Nat → ℝ) (i j : Nat) : ℝ :=
  ∑ k : Fin inner, A i k.val*B k.val j

def blockSum (inner width : Nat) (A B : Nat → Nat → ℝ) (i j block : Nat) : ℝ :=
  ∑ k : Fin width, if block*width+k.val< inner then A i (block*width+k.val)*B (block*width+k.val) j else 0

theorem sum_blocks (f : Nat → ℝ) (blocks width : Nat) :
    (∑ b∈Finset.range blocks, ∑ k : Fin width,f (b*width+k.val))=
      ∑ k∈Finset.range (blocks*width),f k := by
  induction blocks with
  | zero => simp
  | succ b ih =>
    rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]
    rw [Fin.sum_univ_eq_sum_range (fun k => f (b*width+k)) width]

theorem masked_blocks_matmul (inner width : Nat) (hw : 0<width) (A B : Nat → Nat → ℝ) (i j : Nat) :
    (∑ b∈Finset.range ((inner+width-1)/width),blockSum inner width A B i j b)=
      mathematical inner A B i j := by
  unfold blockSum
  rw [sum_blocks (fun k => if k< inner then A i k*B k j else 0)]
  let f := fun k => if k< inner then A i k*B k j else 0
  change (∑k∈Finset.range (((inner+width-1)/width)*width),f k)=_
  have ht : (∑k∈Finset.range (((inner+width-1)/width)*width),f k)=∑k∈Finset.range inner,f k := by
    symm
    apply Finset.sum_subset (Finset.range_mono (ceil_cover inner width hw))
    intro k _ hk
    simp only [Finset.mem_range] at hk
    simp [f,hk]
  rw [ht,Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  simp [f,k.isLt,mathematical]

/-- Finite FP32 projection. Nonfinite words are excluded by the numerical
boundary below; zero here only makes the function total. -/
def realValue (v : Word) : ℝ :=
  match v.toRat? with
  | some q => (q.num : ℝ)/(q.den : ℝ)
  | none => 0

def inputA [Inhabited Word] (r : Request) (s : State Word) (i k : Nat) : ℝ :=
  realValue ((s.mem r.A (i*max r.inner 1+k)).read .fp32)
def inputB [Inhabited Word] (r : Request) (s : State Word) (k j : Nat) : ℝ :=
  realValue ((s.mem r.B (k*max r.columns 1+j)).read .fp32)

inductive Reached [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (p : Parameters) (s : State Word) : Nat → Accumulator p → Prop
  | initial : Reached cfg D F g p s 0 (fun _ => FP.Value.zero .fp32)
  | next {k a} : Reached cfg D F g p s k a → k<count p →
      Reached cfg D F g p s (k+1) (update cfg D F g p s k a)

/-- Explicit, deferred numerical obligations at the actual reached primitive
calls. `step` concerns only FP dot/add (or DotAcc) arithmetic on loaded tiles;
loads, addresses, loop bounds, stores, grid and return are proved separately.
This is a conditional exact Real projection, not a claim that arbitrary FP32
inputs have exact dot accumulation. Terminal half narrowing remains explicit. -/
structure NumericalBoundary [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s : State Word) (roundHalf : ℝ → ℝ) : Prop where
  zero : realValue (FP.Value.zero .fp32)=0
  finite : ∀ pid k a, Reached cfg D F g r.parameters (instanceState pid s) k a →
    ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true → (a q).toRat?.isSome=Bool.true
  step : ∀ pid k a, Reached cfg D F g r.parameters (instanceState pid s) k a → k<count r.parameters →
    ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
    realValue (update cfg D F g r.parameters (instanceState pid s) k a q)=realValue (a q)+
      ∑ j : Fin r.blockInner,
        realValue (aValues r.parameters (instanceState pid s) k (q.1,j,PUnit.unit))*
        realValue (bValues r.parameters (instanceState pid s) k (j,q.2.1,PUnit.unit))
  narrowing : ∀ pid a, Reached cfg D F g r.parameters (instanceState pid s) (count r.parameters) a →
    ∀ q : CIndex r, active r.parameters (instanceState pid s) q=Bool.true →
      realValue (outputValues cfg r.parameters a q)=roundHalf (realValue (a q))

theorem loaded_block [Inhabited Word] (r : Request) (s : State Word) (pid k : Nat) (q : CIndex r)
    (hq : active r.parameters (instanceState pid s) q=Bool.true)
    (hz : realValue (FP.Value.zero .fp32)=0) :
    (∑ j : Fin r.blockInner,
      realValue (aValues r.parameters (instanceState pid s) k (q.1,j,PUnit.unit))*
      realValue (bValues r.parameters (instanceState pid s) k (j,q.2.1,PUnit.unit)))=
      blockSum r.inner r.blockInner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid s) q.1) (col r.parameters (instanceState pid s) q.2.1) k := by
  have hq' : row r.parameters (instanceState pid s) q.1<r.rows ∧
      col r.parameters (instanceState pid s) q.2.1<r.columns := by
    simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hq
  unfold blockSum
  apply Finset.sum_congr rfl
  intro j _
  have hguard : (j.val<r.inner-k*r.blockInner) ↔ k*r.blockInner+j.val<r.inner := by omega
  have ha : aOffset r.parameters (instanceState pid s) k (q.1,j,PUnit.unit)=
      row r.parameters (instanceState pid s) q.1*max r.inner 1+(k*r.blockInner+j.val) := by
    simp only [aOffset,aBase]
    change (row r.parameters (instanceState pid s) q.1%r.rows)*max r.inner 1+ j.val*1+k*r.blockInner*1=_
    rw [Nat.mod_eq_of_lt hq'.1]
    ring
  have hb : bOffset r.parameters (instanceState pid s) k (j,q.2.1,PUnit.unit)=
      (k*r.blockInner+j.val)*max r.columns 1+col r.parameters (instanceState pid s) q.2.1 := by
    simp only [bOffset,bBase]
    change j.val*max r.columns 1+(col r.parameters (instanceState pid s) q.2.1%r.columns)*1+k*r.blockInner*max r.columns 1=_
    rw [Nat.mod_eq_of_lt hq'.2]
    ring
  change realValue (if j.val<r.inner-k*r.blockInner then
      ((instanceState pid s).mem r.A (aOffset r.parameters (instanceState pid s) k (q.1,j,PUnit.unit))).read .fp32
      else FP.Value.zero .fp32)*
    realValue (if j.val<r.inner-k*r.blockInner then
      ((instanceState pid s).mem r.B (bOffset r.parameters (instanceState pid s) k (j,q.2.1,PUnit.unit))).read .fp32
      else FP.Value.zero .fp32)=_
  rw [ha,hb]
  by_cases hk : k*r.blockInner+j.val<r.inner
  · simp [hguard.mpr hk,hk,inputA,inputB,instanceState]
  · have hn : ¬j.val<r.inner-k*r.blockInner := by omega
    simp [hn,hk,hz]

theorem range_projection [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s : State Word) (roundHalf : ℝ → ℝ) (law : NumericalBoundary cfg D F g r s roundHalf)
    (pid cur : Nat) (a : Accumulator r.parameters)
    (hr : Reached cfg D F g r.parameters (instanceState pid s) cur a) (hc : cur≤count r.parameters)
    (q : CIndex r) (hq : active r.parameters (instanceState pid s) q=Bool.true) :
    Reached cfg D F g r.parameters (instanceState pid s) (count r.parameters)
      (foldRange (update cfg D F g r.parameters (instanceState pid s)) cur (count r.parameters) 1 a) ∧
    realValue (foldRange (update cfg D F g r.parameters (instanceState pid s)) cur (count r.parameters) 1 a q)=
      realValue (a q)+
      (∑ k∈Finset.range (count r.parameters),blockSum r.inner r.blockInner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid s) q.1) (col r.parameters (instanceState pid s) q.2.1) k)-
      (∑ k∈Finset.range cur,blockSum r.inner r.blockInner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid s) q.1) (col r.parameters (instanceState pid s) q.2.1) k) := by
  by_cases hs : cur<count r.parameters
  · rw [foldRange]
    simp only [Nat.one_ne_zero,hs,↓reduceDIte]
    have hh := range_projection cfg D F g r s roundHalf law pid (cur+1)
      (update cfg D F g r.parameters (instanceState pid s) cur a) (Reached.next hr hs) (by omega) q hq
    refine ⟨hh.1,?_⟩
    rw [hh.2,law.step pid cur a hr hs q hq,loaded_block r s pid cur q hq law.zero,Finset.sum_range_succ]
    ring
  · have he : cur=count r.parameters := by omega
    subst cur
    rw [foldRange]
    simp only [Nat.one_ne_zero,lt_self_iff_false,↓reduceDIte]
    exact ⟨hr,by ring⟩
termination_by count r.parameters-cur
decreasing_by omega

theorem program_projection [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s : State Word) (roundHalf : ℝ → ℝ) (law : NumericalBoundary cfg D F g r s roundHalf)
    (pid : Nat) (q : CIndex r) (hq : active r.parameters (instanceState pid s) q=Bool.true) :
    realValue (outputValues cfg r.parameters (accumulated cfg D F g r.parameters (instanceState pid s)) q)=
      roundHalf (mathematical r.inner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid s) q.1) (col r.parameters (instanceState pid s) q.2.1)) := by
  have hh := range_projection cfg D F g r s roundHalf law pid 0 (fun _ => FP.Value.zero .fp32)
    Reached.initial (Nat.zero_le _) q hq
  unfold accumulated
  rw [law.narrowing pid _ hh.1 q hq]
  have hm := hh.2
  simp only [Finset.range_zero,Finset.sum_empty,sub_zero,law.zero,zero_add] at hm
  rw [hm]
  congr 1
  exact masked_blocks_matmul r.inner r.blockInner r.positiveInnerBlock _ _ _ _

theorem reference_projection [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s : State Word) (roundHalf : ℝ → ℝ) (law : NumericalBoundary cfg D F g r s roundHalf)
    (i j : Nat) (hi : i<r.rows) (hj : j<r.columns) :
    realValue (reference cfg D F g r s i j)=roundHalf (mathematical r.inner (inputA r s) (inputB r s) i j) := by
  obtain ⟨pid,_,q,hrow,hcol⟩ := cell_covered r s i j hi hj
  have hq : active r.parameters (instanceState pid s) q=Bool.true := by
    simp only [active,Bool.and_eq_true,decide_eq_true_eq]
    change row r.parameters (instanceState pid s) q.1<r.rows ∧ col r.parameters (instanceState pid s) q.2.1<r.columns
    rw [hrow,hcol]
    exact ⟨hi,hj⟩
  have hp := program_projection cfg D F g r s roundHalf law pid q hq
  rw [source_reference cfg D F g r s s pid (fun _ => rfl) (fun _ => rfl) q] at hp
  simpa only [hrow,hcol] using hp

theorem host_projection [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (s : State Word) (a : Allocation r s) (roundHalf : ℝ → ℝ)
    (law : NumericalBoundary cfg D F g r s roundHalf) :
    ∃ final, host rest cfg D F g r s=some (outputMetadata r,final) ∧
      (∀ i<r.rows, ∀ j<r.columns,
        realValue ((final.mem r.C (offset r i j)).read .fp32)=
          roundHalf (mathematical r.inner (inputA r s) (inputB r s) i j)) ∧
      (∀ region o, ¬Footprint r region o → final.mem region o=s.mem region o) := by
  obtain ⟨final,he,hv,hf⟩ := host_correct rest cfg D F g r s a
  refine ⟨final,he,?_,hf⟩
  intro i hi j hj
  rw [hv i hi j hj,Cell.read_same]
  exact reference_projection cfg D F g r s roundHalf law i j hi hj

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.RealTarget
