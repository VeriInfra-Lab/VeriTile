import bench.tritonbench_g.iv_dependent_matmul.Proofs.HostSource
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Schedules
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Active
import Mathlib.Tactic

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
open VeriTile.Triton FP.Structural FPValue Kernels
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

theorem ceil_cover (n b : Nat) (hb : 0<b) : n≤((n+b-1)/b)*b := by
  have hmod := Nat.mod_lt (n+b-1) hb
  have hdiv := Nat.div_add_mod' (n+b-1) b
  omega

theorem coordinate_block_lt (i n b : Nat) (hb : 0<b) (hi : i<n) : i/b<(n+b-1)/b := by
  apply (Nat.div_lt_iff_lt_mul hb).mpr
  exact lt_of_lt_of_le hi (ceil_cover n b hb)

theorem source_offset (r : Request) (s : State Word) (q : CIndex r) :
    outputOffset r.parameters s q=offset r (row r.parameters s q.1) (col r.parameters s q.2.1) := by
  simp [outputOffset,offset,Request.parameters]

theorem offset_injective (r : Request) {i j i' j' : Nat} (hj : j<r.columns) (hj' : j'<r.columns)
    (h : offset r i j=offset r i' j') : i=i' ∧ j=j' := by
  have hn : 0<max r.columns 1 := by omega
  have hjn : j<max r.columns 1 := lt_of_lt_of_le hj (le_max_left _ _)
  have hjn' : j'<max r.columns 1 := lt_of_lt_of_le hj' (le_max_left _ _)
  unfold offset at h
  have hm := congrArg (fun n => n%max r.columns 1) h
  simp [Nat.add_mod,Nat.mod_eq_of_lt hjn,Nat.mod_eq_of_lt hjn'] at hm
  subst j'
  exact ⟨Nat.eq_of_mul_eq_mul_left hn (Nat.add_right_cancel h),rfl⟩

theorem output_bound (r : Request) (i j : Nat) (hi : i<r.rows) (hj : j<r.columns) :
    offset r i j<r.rows*r.columns := by
  have hn : max r.columns 1=r.columns := by omega
  unfold offset
  rw [hn]
  have hh := Nat.mul_le_mul_left r.columns (show i+1≤r.rows by omega)
  nlinarith

theorem cell_covered (r : Request) (s : State Word) (i j : Nat) (hi : i<r.rows) (hj : j<r.columns) :
    Covered r s (List.range (programs r)) i j := by
  have hir := coordinate_block_lt i r.rows r.blockRows r.positiveRowsBlock hi
  have hjc : j/r.blockColumns<tileColumns r := coordinate_block_lt j r.columns r.blockColumns r.positiveColumnsBlock hj
  have hcn : 0<tileColumns r := Nat.zero_lt_of_lt hjc
  have hp : canonicalPid r i j<programs r := by
    have hm := Nat.mul_le_mul_right (tileColumns r) (Nat.succ_le_of_lt hir)
    unfold canonicalPid programs tileRows tileColumns at *
    nlinarith
  refine ⟨canonicalPid r i j,List.mem_range.mpr hp,canonicalIndex r i j,?_,?_⟩
  · simp only [row,rowBlock,columnsGrid,instanceState,Request.parameters,canonicalPid,canonicalIndex]
    change ((i/r.blockRows*tileColumns r+j/r.blockColumns)/tileColumns r)*r.blockRows+i%r.blockRows=i
    have hd : (i/r.blockRows*tileColumns r+j/r.blockColumns)/tileColumns r=i/r.blockRows := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ hcn,Nat.div_eq_of_lt hjc,Nat.zero_add]
    rw [hd]
    exact Nat.div_add_mod' i r.blockRows
  · simp only [col,colBlock,columnsGrid,instanceState,Request.parameters,canonicalPid,canonicalIndex]
    change ((i/r.blockRows*tileColumns r+j/r.blockColumns)%tileColumns r)*r.blockColumns+j%r.blockColumns=j
    rw [Nat.mul_add_mod',Nat.mod_eq_of_lt hjc]
    exact Nat.div_add_mod' j r.blockColumns

theorem canonical_of_lane (r : Request) (s : State Word) (pid : Nat) (q : CIndex r) :
    canonicalPid r (row r.parameters (instanceState pid s) q.1)
        (col r.parameters (instanceState pid s) q.2.1)=pid ∧
      canonicalIndex r (row r.parameters (instanceState pid s) q.1)
        (col r.parameters (instanceState pid s) q.2.1)=q := by
  have hrow : row r.parameters (instanceState pid s) q.1/r.blockRows=pid/tileColumns r := by
    change ((pid/tileColumns r)*r.blockRows+q.1.val)/r.blockRows=_
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ r.positiveRowsBlock,Nat.div_eq_of_lt q.1.isLt,Nat.zero_add]
  have hcol : col r.parameters (instanceState pid s) q.2.1/r.blockColumns=pid%tileColumns r := by
    change ((pid%tileColumns r)*r.blockColumns+q.2.1.val)/r.blockColumns=_
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ r.positiveColumnsBlock,Nat.div_eq_of_lt q.2.1.isLt,Nat.zero_add]
  constructor
  · unfold canonicalPid
    rw [hrow,hcol]
    exact Nat.div_add_mod' pid (tileColumns r)
  · have hr : row r.parameters (instanceState pid s) q.1%r.blockRows=q.1.val := by
      change ((pid/tileColumns r)*r.blockRows+q.1.val)%r.blockRows=_
      exact Nat.mul_add_mod_of_lt q.1.isLt
    have hc : col r.parameters (instanceState pid s) q.2.1%r.blockColumns=q.2.1.val := by
      change ((pid%tileColumns r)*r.blockColumns+q.2.1.val)%r.blockColumns=_
      exact Nat.mul_add_mod_of_lt q.2.1.isLt
    apply Prod.ext
    · exact Fin.ext hr
    · apply Prod.ext
      · exact Fin.ext hc
      · rfl

theorem accumulated_inputs [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial current : State Word) (pid : Nat)
    (ha : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hb : ∀ o, current.mem r.B o=initial.mem r.B o) :
    accumulated cfg D F g r.parameters (instanceState pid current)=
      accumulated cfg D F g r.parameters (instanceState pid initial) := by
  have hav (k : Nat) : aValues r.parameters (instanceState pid current) k=
      aValues r.parameters (instanceState pid initial) k := by
    funext q
    simp only [aValues,instanceState,Request.parameters,aOffset,aBase,row,rowBlock,columnsGrid]
    rw [ha]
  have hbv (k : Nat) : bValues r.parameters (instanceState pid current) k=
      bValues r.parameters (instanceState pid initial) k := by
    funext q
    simp only [bValues,instanceState,Request.parameters,bOffset,bBase,col,colBlock,columnsGrid]
    rw [hb]
  have hu : update cfg D F g r.parameters (instanceState pid current)=
      update cfg D F g r.parameters (instanceState pid initial) := by
    funext k a
    cases g <;> simp only [update,hav,hbv]
  unfold accumulated
  rw [hu]

theorem source_reference [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial current : State Word) (pid : Nat)
    (ha : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hb : ∀ o, current.mem r.B o=initial.mem r.B o) (q : CIndex r) :
    outputValues cfg r.parameters (accumulated cfg D F g r.parameters (instanceState pid current)) q=
      reference cfg D F g r initial (row r.parameters (instanceState pid current) q.1)
        (col r.parameters (instanceState pid current) q.2.1) := by
  unfold reference
  rw [(canonical_of_lane r current pid q).1,(canonical_of_lane r current pid q).2]
  rw [accumulated_inputs cfg D F g r initial current pid ha hb]

theorem program_correct [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial current : State Word) (pid : Nat)
    (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o) :
    ∃ next, FP.Structural.exec (backend rest cfg D F) (Kernels.source r.schedule g r.parameters)
        (instanceState pid current)=some next ∧
      (∀ o, next.mem r.A o=initial.mem r.A o) ∧
      (∀ o, next.mem r.B o=initial.mem r.B o) ∧
      (∀ region o, ¬Footprint r region o → next.mem region o=current.mem region o) ∧
      (∀ i<r.rows, ∀ j<r.columns, Good cfg D F g r initial current i j → Good cfg D F g r initial next i j) ∧
      (∀ i<r.rows, ∀ j<r.columns, Covered r initial [pid] i j → Good cfg D F g r initial next i j) := by
  let start := instanceState pid current
  obtain ⟨next,he,hm⟩ := Schedules.execution_memory rest cfg D F r.schedule g r.parameters start
  have hp := written_post_active cfg r.parameters start (accumulated cfg D F g r.parameters start)
    (output_injective_active r.parameters start rfl (le_max_left _ _))
  have hp' : Post cfg r.parameters start next (accumulated cfg D F g r.parameters start) := by
    simpa only [Post,hm] using hp
  obtain ⟨hv,hf⟩ := hp'
  have value (q : CIndex r) (hq : active r.parameters start q=Bool.true) :
      Good cfg D F g r initial next (row r.parameters start q.1) (col r.parameters start q.2.1) := by
    have h := hv q hq
    rw [source_reference cfg D F g r initial current pid hA hB q,source_offset] at h
    exact h
  refine ⟨next,he,?_,?_,?_,?_,?_⟩
  · intro o
    exact (hf r.A o (Or.inl (Ne.symm hCA))).trans (hA o)
  · intro o
    exact (hf r.B o (Or.inl (Ne.symm hCB))).trans (hB o)
  · intro region o hn
    apply hf
    by_cases hr : region=r.C
    · right
      intro q hq heq
      apply hn
      have hq' : row r.parameters start q.1<r.rows ∧ col r.parameters start q.2.1<r.columns := by
        simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hq
      refine ⟨hr,?_⟩
      rw [heq,source_offset]
      exact output_bound r _ _ hq'.1 hq'.2
    · exact Or.inl hr
  · intro i hi j hj hg
    by_cases hw : ∃ q : CIndex r, active r.parameters start q=Bool.true ∧ offset r i j=outputOffset r.parameters start q
    · obtain ⟨q,hq,hw⟩ := hw
      have hq' : row r.parameters start q.1<r.rows ∧ col r.parameters start q.2.1<r.columns := by
        simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hq
      rw [source_offset] at hw
      have hh := offset_injective r hj hq'.2 hw
      simpa only [hh.1,hh.2] using value q hq
    · exact (hf r.C (offset r i j) (Or.inr (by intro q hq heq; exact hw ⟨q,hq,heq⟩))).trans hg
  · intro i hi j hj hcov
    obtain ⟨p,hp,q,hrow,hcol⟩ := hcov
    have hp' : p=pid := by simpa using hp
    subst p
    have hrow' : row r.parameters start q.1=i := hrow
    have hcol' : col r.parameters start q.2.1=j := hcol
    have hq : active r.parameters start q=Bool.true := by
      simp only [active,Bool.and_eq_true,decide_eq_true_eq]
      change row r.parameters start q.1<r.rows ∧ col r.parameters start q.2.1<r.columns
      rw [hrow',hcol']
      exact ⟨hi,hj⟩
    simpa only [hrow',hcol'] using value q hq

theorem programs_correct [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial current : State Word) (ids : List Nat)
    (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o) :
    ∃ final, runPrograms rest cfg D F g r ids current=some final ∧
      (∀ o, final.mem r.A o=initial.mem r.A o) ∧
      (∀ o, final.mem r.B o=initial.mem r.B o) ∧
      (∀ region o, ¬Footprint r region o → final.mem region o=current.mem region o) ∧
      (∀ i<r.rows, ∀ j<r.columns, Good cfg D F g r initial current i j → Good cfg D F g r initial final i j) ∧
      (∀ i<r.rows, ∀ j<r.columns, Covered r initial ids i j → Good cfg D F g r initial final i j) := by
  induction ids generalizing current with
  | nil =>
    refine ⟨current,rfl,hA,hB,fun _ _ _ => rfl,fun _ _ _ _ h => h,?_⟩
    intro i hi j hj hcov
    obtain ⟨p,hp,_⟩ := hcov
    simp at hp
  | cons pid ids ih =>
    obtain ⟨next,he,ha,hb,hframe,hkeep,hvalue⟩ := program_correct rest cfg D F g r initial current pid hCA hCB hA hB
    obtain ⟨final,hfin,ha',hb',hframe',hkeep',hvalue'⟩ := ih next ha hb
    refine ⟨final,?_,ha',hb',?_,?_,?_⟩
    · simp only [runPrograms,he,Option.bind,bind,hfin]
    · intro region o hn
      exact (hframe' region o hn).trans (hframe region o hn)
    · intro i hi j hj hg
      exact hkeep' i hi j hj (hkeep i hi j hj hg)
    · intro i hi j hj hcov
      obtain ⟨p,hp,q,hrow,hcol⟩ := hcov
      rcases List.mem_cons.mp hp with hp | hp
      · subst p
        exact hkeep' i hi j hj (hvalue i hi j hj ⟨pid,by simp,q,hrow,hcol⟩)
      · exact hvalue' i hi j hj ⟨p,hp,q,hrow,hcol⟩

theorem grid_correct [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial : State Word) (hCA : r.C≠r.A) (hCB : r.C≠r.B) :
    ∃ final, runPrograms rest cfg D F g r (List.range (programs r)) initial=some final ∧
      Result cfg D F g r initial final := by
  obtain ⟨final,he,_,_,hf,_,hv⟩ := programs_correct rest cfg D F g r initial initial
    (List.range (programs r)) hCA hCB (fun _ => rfl) (fun _ => rfl)
  exact ⟨final,he,fun i hi j hj => hv i hi j hj (cell_covered r initial i j hi hj),hf⟩

theorem host_correct [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial : State Word) (a : Allocation r initial) :
    ∃ final, host rest cfg D F g r initial=some (outputMetadata r,final) ∧ Result cfg D F g r initial final := by
  obtain ⟨final,he,hp⟩ := grid_correct rest cfg D F g r initial a.freshA a.freshB
  exact ⟨final,by simp [host,he],hp⟩

theorem grid_order_correct [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial : State Word) (a : Allocation r initial)
    (ids : List Nat) (order : ids.Perm (List.range (programs r))) :
    ∃ final, runPrograms rest cfg D F g r ids initial=some final ∧ Result cfg D F g r initial final := by
  obtain ⟨final,he,_,_,hf,_,hv⟩ := programs_correct rest cfg D F g r initial initial
    ids a.freshA a.freshB (fun _ => rfl) (fun _ => rfl)
  refine ⟨final,he,?_,hf⟩
  intro i hi j hj
  obtain ⟨pid,hp,q,hrow,hcol⟩ := cell_covered r initial i j hi hj
  exact hv i hi j hj ⟨pid,order.mem_iff.mpr hp,q,hrow,hcol⟩

theorem source_store_unique (r : Request) (s : State Word) (p q : Nat)
    (i j : CIndex r)
    (hi : active r.parameters (instanceState p s) i=Bool.true)
    (hj : active r.parameters (instanceState q s) j=Bool.true)
    (he : outputOffset r.parameters (instanceState p s) i=outputOffset r.parameters (instanceState q s) j) :
    p=q ∧ i=j := by
  have hi' : row r.parameters (instanceState p s) i.1<r.rows ∧ col r.parameters (instanceState p s) i.2.1<r.columns := by
    simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hi
  have hj' : row r.parameters (instanceState q s) j.1<r.rows ∧ col r.parameters (instanceState q s) j.2.1<r.columns := by
    simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hj
  rw [source_offset,source_offset] at he
  have hh := offset_injective r hi'.2 hj'.2 he
  have hp := canonical_of_lane r s p i
  have hq := canonical_of_lane r s q j
  rw [hh.1,hh.2] at hp
  exact ⟨hp.1.symm.trans hq.1,hp.2.symm.trans hq.2⟩

theorem result_memory_unique [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s x y : State Word) (hx : Result cfg D F g r s x) (hy : Result cfg D F g r s y) : x.mem=y.mem := by
  funext region o
  by_cases hf : Footprint r region o
  · obtain ⟨hr,ho⟩ := hf
    subst region
    have hn : 0<r.columns := by nlinarith
    have hi : o/r.columns<r.rows := (Nat.div_lt_iff_lt_mul hn).mpr ho
    have hj : o%r.columns<r.columns := Nat.mod_lt _ hn
    have he : offset r (o/r.columns) (o%r.columns)=o := by
      unfold offset
      rw [max_eq_left (show 1≤r.columns by omega)]
      exact Nat.div_add_mod o r.columns
    have h1 := hx.1 _ hi _ hj
    have h2 := hy.1 _ hi _ hj
    unfold Good at h1 h2
    rw [he] at h1 h2
    exact h1.trans h2.symm
  · exact (hx.2 region o hf).trans (hy.2 region o hf).symm

theorem host_schedule_equivalence [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (g : Graph) (r : Request) (s : State Word)
    (allocation : Allocation r s) (a b : Schedule) :
    ∃ x y, host rest cfg D F g {r with schedule:=a} s=some (outputMetadata r,x) ∧
      host rest cfg D F g {r with schedule:=b} s=some (outputMetadata r,y) ∧ x.mem=y.mem := by
  obtain ⟨x,hx,hpx⟩ := grid_correct rest cfg D F g {r with schedule:=a} s allocation.freshA allocation.freshB
  obtain ⟨y,hy,hpy⟩ := grid_correct rest cfg D F g {r with schedule:=b} s allocation.freshA allocation.freshB
  exact ⟨x,y,by simp [host,hx,outputMetadata],by simp [host,hy,outputMetadata],result_memory_unique cfg D F g r s x y hpx hpy⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
