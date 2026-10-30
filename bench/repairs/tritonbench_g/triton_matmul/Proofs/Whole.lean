import bench.repairs.tritonbench_g.triton_matmul.Proofs.Execution
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Geometry
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Transfer the existing complete per-program readback/frame theorem using
the newly proved actual candidate execution. No successful-run input. -/
theorem candidate_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (initial : State α)
    (hw : ∀ i j, FPValue.active r.parameters initial i → FPValue.active r.parameters initial j →
      FPValue.cOffset r.parameters initial i=FPValue.cOffset r.parameters initial j → i=j) :
    ∃ final, FP.Structural.exec (FPValue.backend M D) (Source.typed r.parameters r.storage) initial=some final ∧
      FPValue.Post r.parameters r.storage initial final (FPValue.outputValue M D r.parameters r.storage initial) := by
  obtain ⟨reference,_,hm,hvalue,hframe⟩ := FPValue.correctness M D r.parameters r.storage initial hw
  obtain ⟨final,he,hf⟩ := Source.execution_memory M D r initial
  refine ⟨final,he,?_,?_⟩
  · simpa only [hf,←hm] using hvalue
  · simpa only [hf,←hm] using hframe

theorem source_block {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (r : Request) (law : PrimitiveLaws M D Q r.storage) (initial current : State α)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o)
    (i : FPValue.CIndex r.parameters) (hi : FPValue.active r.parameters current i)
    (block : Nat) (acc : FPValue.CIndex r.parameters → α) :
    FPValue.update M D r.parameters r.storage current block acc i=
      blockStep Q r initial
        (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) block (acc i) := by
  have haRow : FPValue.aRow r.parameters current i.1=FPValue.row r.parameters current i.1 :=
    by simp [FPValue.aRow,hi.1]
  have hbCol : FPValue.bCol r.parameters current i.2.1=FPValue.col r.parameters current i.2.1 :=
    by simp [FPValue.bCol,hi.2]
  have ha (j : Fin (SourceScope.blockK r.storage)) :
      FPValue.aValue M r.parameters r.storage current block (i.1,j,PUnit.unit)=
        (if block*(SourceScope.blockK r.storage)+j.val<r.inner then
          inputA r initial (FPValue.row r.parameters current i.1) (block*(SourceScope.blockK r.storage)+j.val) else Q.loadZero) := by
    have he : FPValue.aOffset r.parameters current block (i.1,j,PUnit.unit)=
        FPValue.row r.parameters current i.1*r.strideAM+(block*(SourceScope.blockK r.storage)+j.val)*r.strideAK := by
      unfold FPValue.aOffset
      rw [haRow]
      simp only [Request.parameters,SourceScope.parameters,FPValue.BK]
      ring
    by_cases hk : block*(SourceScope.blockK r.storage)+j.val<r.inner
    · have hj : j.val<r.inner-block*(SourceScope.blockK r.storage) := by omega
      simp only [FPValue.aValue,Request.parameters,SourceScope.parameters,FPValue.BK,hj,if_true]
      change FPValue.read r.storage (current.mem r.A (FPValue.aOffset r.parameters current block (i.1,j,PUnit.unit)))=_
      rw [he,hA]
      simp [inputA,hk]
      rfl
    · have hj : ¬j.val<r.inner-block*(SourceScope.blockK r.storage) := by omega
      simp only [FPValue.aValue,Request.parameters,SourceScope.parameters,FPValue.BK,hj,if_false]
      simpa [hk,FPValue.dtype] using law.loadZero
  have hb (j : Fin (SourceScope.blockK r.storage)) :
      FPValue.bValue M r.parameters r.storage current block (j,i.2.1,PUnit.unit)=
        (if block*(SourceScope.blockK r.storage)+j.val<r.inner then
          inputB r initial (block*(SourceScope.blockK r.storage)+j.val) (FPValue.col r.parameters current i.2.1) else Q.loadZero) := by
    have he : FPValue.bOffset r.parameters current block (j,i.2.1,PUnit.unit)=
        (block*(SourceScope.blockK r.storage)+j.val)*r.strideBK+FPValue.col r.parameters current i.2.1*r.strideBN := by
      unfold FPValue.bOffset
      rw [hbCol]
      simp only [Request.parameters,SourceScope.parameters,FPValue.BK]
      ring
    by_cases hk : block*(SourceScope.blockK r.storage)+j.val<r.inner
    · have hj : j.val<r.inner-block*(SourceScope.blockK r.storage) := by omega
      simp only [FPValue.bValue,Request.parameters,SourceScope.parameters,FPValue.BK,hj,if_true]
      change FPValue.read r.storage (current.mem r.B (FPValue.bOffset r.parameters current block (j,i.2.1,PUnit.unit)))=_
      rw [he,hB]
      simp [inputB,hk]
      rfl
    · have hj : ¬j.val<r.inner-block*(SourceScope.blockK r.storage) := by omega
      simp only [FPValue.bValue,Request.parameters,SourceScope.parameters,FPValue.BK,hj,if_false]
      simpa [hk,FPValue.dtype] using law.loadZero
  unfold FPValue.update
  change D.apply (some .fp32) (FPValue.dtype r.storage) (FPValue.dtype r.storage) .real {} (batch := [])
    (m := 128) (k := (SourceScope.blockK r.storage)) (n := 256) _ _ acc i=_
  rw [law.dotAcc]
  unfold blockStep
  apply congrArg₂ (fun a b => Q.dotAcc (SourceScope.blockK r.storage) a b (acc i))
  · funext j; exact ha j
  · funext j; exact hb j

theorem source_reference {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (r : Request) (law : PrimitiveLaws M D Q r.storage) (initial current : State α)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o)
    (i : FPValue.CIndex r.parameters) (hi : FPValue.active r.parameters current i) :
    FPValue.outputValue M D r.parameters r.storage current i=
      reference Q r initial (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) := by
  have hacc : FPValue.accumulation M D r.parameters r.storage current i=
      accumulator Q r initial (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) := by
    unfold FPValue.accumulation accumulator
    rw [MatmulTritonAutotune.Whole.foldRange_pointwise _ (fun block x => blockStep Q r initial
      (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) block x)
      i (source_block M D Q r law initial current hA hB i hi)]
    rw [law.initialZero]
  unfold FPValue.outputValue reference
  rw [hacc,law.outputCast]

def Good {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (initial current : State α) (row col : Nat) : Prop :=
  current.mem r.C (outputOffset r row col)=FPValue.stored r.storage (reference Q r initial row col)

def Covered {α : Type} (r : Request) (s : State α) (ids : List Nat) (row col : Nat) : Prop :=
  ∃ pid∈ids, ∃ i : FPValue.CIndex r.parameters,
    FPValue.row r.parameters (instanceState r pid s) i.1=row ∧
    FPValue.col r.parameters (instanceState r pid s) i.2.1=col

theorem source_offset {α : Type} (r : Request) (s : State α) (i : FPValue.CIndex r.parameters) :
    FPValue.cOffset r.parameters s i=
      outputOffset r (FPValue.row r.parameters s i.1) (FPValue.col r.parameters s i.2.1) := by
  simp [FPValue.cOffset,outputOffset,Request.parameters,SourceScope.parameters]

theorem program_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (r : Request) (law : PrimitiveLaws M D Q r.storage) (initial current : State α) (pid : Nat)
    (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o) :
    ∃ next, FP.Structural.exec (FPValue.backend M D) (kernel r) (instanceState r pid current)=some next ∧
      (∀ o, next.mem r.A o=initial.mem r.A o) ∧
      (∀ o, next.mem r.B o=initial.mem r.B o) ∧
      (∀ region offset, ¬Footprint r region offset → next.mem region offset=current.mem region offset) ∧
      (∀ row<r.rows, ∀ col<r.columns, Good Q r initial current row col → Good Q r initial next row col) ∧
      (∀ row<r.rows, ∀ col<r.columns, Covered r initial [pid] row col → Good Q r initial next row col) := by
  let start := instanceState r pid current
  obtain ⟨next,he,hv,hf⟩ := candidate_correctness M D r start
    (active_injective r start)
  have value (i : FPValue.CIndex r.parameters) (hi : FPValue.active r.parameters start i) :
      Good Q r initial next (FPValue.row r.parameters start i.1) (FPValue.col r.parameters start i.2.1) := by
    have h := hv i hi
    rw [source_reference M D Q r law initial start hA hB i hi,source_offset] at h
    exact h
  refine ⟨next,he,?_,?_,?_,?_,?_⟩
  · intro o
    exact (hf r.A o (Or.inl (Ne.symm hCA))).trans (hA o)
  · intro o
    exact (hf r.B o (Or.inl (Ne.symm hCB))).trans (hB o)
  · intro region offset hn
    apply hf
    by_cases hr : region=r.C
    · right
      intro i hi heq
      apply hn
      refine ⟨hr,FPValue.row r.parameters start i.1,hi.1,
        FPValue.col r.parameters start i.2.1,hi.2,?_⟩
      exact heq.trans (source_offset r start i)
    · exact Or.inl hr
  · intro row hr col hc hg
    by_cases hw : ∃ i : FPValue.CIndex r.parameters,
        FPValue.active r.parameters start i ∧ outputOffset r row col=FPValue.cOffset r.parameters start i
    · obtain ⟨i,hi,hw⟩ := hw
      rw [source_offset] at hw
      have hh := offset_injective r hc hi.2 hw
      simpa only [hh.1,hh.2] using value i hi
    · have hframe := hf r.C (outputOffset r row col) (Or.inr (by
        intro i hi heq
        exact hw ⟨i,hi,heq⟩))
      exact hframe.trans hg
  · intro row hr col hc hcov
    obtain ⟨p,hp,i,hrow,hcol⟩ := hcov
    have hp' : p=pid := by simpa using hp
    subst p
    have hrow' : FPValue.row r.parameters start i.1=row := hrow
    have hcol' : FPValue.col r.parameters start i.2.1=col := hcol
    have hi : FPValue.active r.parameters start i := ⟨by rw [hrow']; exact hr,by rw [hcol']; exact hc⟩
    simpa only [hrow',hcol'] using value i hi

theorem programs_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (r : Request) (law : PrimitiveLaws M D Q r.storage) (initial current : State α) (ids : List Nat)
    (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o) :
    ∃ final, runPrograms M D r ids current=some final ∧
      (∀ o, final.mem r.A o=initial.mem r.A o) ∧
      (∀ o, final.mem r.B o=initial.mem r.B o) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=current.mem region offset) ∧
      (∀ row<r.rows, ∀ col<r.columns, Good Q r initial current row col → Good Q r initial final row col) ∧
      (∀ row<r.rows, ∀ col<r.columns, Covered r initial ids row col → Good Q r initial final row col) := by
  induction ids generalizing current with
  | nil =>
    refine ⟨current,rfl,hA,hB,fun _ _ _ => rfl,fun _ _ _ _ h => h,?_⟩
    intro row hr col hc hcov
    obtain ⟨p,hp,_⟩ := hcov
    simp at hp
  | cons pid ids ih =>
    obtain ⟨next,he,ha,hb,hframe,hkeep,hvalue⟩ := program_correct M D Q r law initial current pid hCA hCB hA hB
    obtain ⟨final,hfin,ha',hb',hframe',hkeep',hvalue'⟩ := ih next ha hb
    refine ⟨final,?_,ha',hb',?_,?_,?_⟩
    · simp only [runPrograms,he,Option.bind_some,hfin]
    · intro region offset hn
      exact (hframe' region offset hn).trans (hframe region offset hn)
    · intro row hr col hc hg
      exact hkeep' row hr col hc (hkeep row hr col hc hg)
    · intro row hr col hc hcov
      obtain ⟨p,hp,i,hi,hj⟩ := hcov
      rcases List.mem_cons.mp hp with hp | hp
      · subst p
        exact hkeep' row hr col hc (hvalue row hr col hc ⟨pid,by simp,i,hi,hj⟩)
      · exact hvalue' row hr col hc ⟨p,hp,i,hi,hj⟩

theorem grid_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (r : Request) (law : PrimitiveLaws M D Q r.storage) (initial : State α) (hCA : r.C≠r.A) (hCB : r.C≠r.B) :
    ∃ final, runPrograms M D r (List.range (programs r)) initial=some final ∧ Result Q r initial final := by
  obtain ⟨final,he,_,_,hf,_,hv⟩ := programs_correct M D Q r law initial initial
    (List.range (programs r)) hCA hCB (fun _ => rfl) (fun _ => rfl)
  refine ⟨final,he,?_,hf⟩
  intro row hr col hc
  obtain ⟨pid,hp,i,hi,hj⟩ := cell_covered r initial row col hr hc
  exact hv row hr col hc ⟨pid,List.mem_range.mpr hp,i,hi,hj⟩

/-- Any scheduling order that visits the actual grid gives the same matrix
observations and frame. Per-program inputs are preserved by fresh C; ownership
is proved below, so this composition needs no scheduler-result assumption. -/
theorem grid_order_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (r : Request) (law : PrimitiveLaws M D Q r.storage) (initial : State α) (ids : List Nat)
    (order : ids.Perm (List.range (programs r))) (hCA : r.C≠r.A) (hCB : r.C≠r.B) :
    ∃ final, runPrograms M D r ids initial=some final ∧ Result Q r initial final := by
  obtain ⟨final,he,_,_,hf,_,hv⟩ := programs_correct M D Q r law initial initial
    ids hCA hCB (fun _ => rfl) (fun _ => rfl)
  refine ⟨final,he,?_,hf⟩
  intro row hr col hc
  obtain ⟨pid,hp,i,hi,hj⟩ := cell_covered r initial row col hr hc
  exact hv row hr col hc ⟨pid,order.mem_iff.mpr (List.mem_range.mpr hp),i,hi,hj⟩

theorem host_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (r : Request) (law : PrimitiveLaws M D Q r.storage) (initial : State α) (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hdims : r.inner=r.innerB) (htype : r.storage=r.storageB) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧ Result Q r initial final := by
  obtain ⟨final,he,hp⟩ := grid_correct M D Q r law initial hCA hCB
  exact ⟨final,by simp [host,hdims,htype,he],hp⟩

theorem source_store_unique {α : Type} (r : Request) (s : State α)
    (p q : Nat) (hp : p<programs r) (hq : q<programs r)
    (i j : FPValue.CIndex r.parameters)
    (hi : FPValue.active r.parameters (instanceState r p s) i)
    (hj : FPValue.active r.parameters (instanceState r q s) j)
    (he : FPValue.cOffset r.parameters (instanceState r p s) i=
      FPValue.cOffset r.parameters (instanceState r q s) j) : p=q ∧ i=j := by
  have h := he
  rw [source_offset,source_offset] at h
  have hh := offset_injective r hi.2 hj.2 h
  have hr : tileRow (tileRows r) (tileColumns r) 8 p*128+i.1.val=
      tileRow (tileRows r) (tileColumns r) 8 q*128+j.1.val := by
    simpa only [FPValue.row,Request.parameters,SourceScope.parameters,FPValue.BM,source_pidM] using hh.1
  have hc : tileCol (tileRows r) (tileColumns r) 8 p*256+i.2.1.val=
      tileCol (tileRows r) (tileColumns r) 8 q*256+j.2.1.val := by
    simpa only [FPValue.col,Request.parameters,SourceScope.parameters,FPValue.BN,source_pidN] using hh.2
  have hpq : p=q := grouped_injective (tileRows r) (tileColumns r) 8 p q (by decide) hp hq
    (by have hi' : i.1.val<128 := i.1.isLt; have hj' : j.1.val<128 := j.1.isLt; omega)
    (by have hi' : i.2.1.val<256 := i.2.1.isLt; have hj' : j.2.1.val<256 := j.2.1.isLt; omega)
  subst q
  exact ⟨rfl,active_injective r (instanceState r p s) i j hi hj he⟩


#axiomsClean host_correct
#axiomsClean grid_order_correct
#axiomsClean source_store_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
