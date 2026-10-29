import bench.tritonbench_g.matmul_leakyrelu.Kernels
import Mathlib.Tactic

namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def groupBase (rows cols pid : Nat) := pid / (4*cols)*4
def groupSize (rows cols pid : Nat) := min (rows-groupBase rows cols pid) 4
def tileRow (rows cols pid : Nat) :=
  groupBase rows cols pid + (pid%(4*cols))%groupSize rows cols pid
def tileCol (rows cols pid : Nat) := (pid%(4*cols))/groupSize rows cols pid

theorem grouped_bounds (rows cols pid : Nat) (hp : pid<rows*cols) :
    0<groupSize rows cols pid ∧ tileRow rows cols pid<rows ∧ tileCol rows cols pid<cols := by
  have hc : 0<cols := by nlinarith
  have hmul := Nat.div_mul_le_self pid (4*cols)
  have hb : groupBase rows cols pid<rows := by unfold groupBase; nlinarith
  have hs : 0<groupSize rows cols pid := by unfold groupSize; omega
  have hspan : groupBase rows cols pid+groupSize rows cols pid≤rows := by
    have := min_le_left (rows-groupBase rows cols pid) 4
    unfold groupSize
    omega
  have hrem : pid%(4*cols)<groupSize rows cols pid*cols := by
    unfold groupSize
    by_cases hsmall : rows-groupBase rows cols pid≤4
    · rw [min_eq_left hsmall]
      have hd := Nat.mod_add_div pid (4*cols)
      have hsub := Nat.sub_add_cancel (Nat.le_of_lt hb)
      unfold groupBase at hsub ⊢
      nlinarith
    · rw [min_eq_right (by omega)]
      exact Nat.mod_lt _ (by omega)
  refine ⟨hs,?_,?_⟩
  · have := Nat.mod_lt (pid%(4*cols)) hs
    unfold tileRow
    omega
  · unfold tileCol
    exact (Nat.div_lt_iff_lt_mul hs).mpr (by simpa [Nat.mul_comm] using hrem)

theorem grouped_injective (rows cols a b : Nat)
    (ha : a<rows*cols) (hb : b<rows*cols)
    (hr : tileRow rows cols a=tileRow rows cols b)
    (hc : tileCol rows cols a=tileCol rows cols b) : a=b := by
  have group_of_row (pid : Nat) (hp : pid<rows*cols) :
      tileRow rows cols pid/4=pid/(4*cols) := by
    have hs := (grouped_bounds rows cols pid hp).1
    have hm := Nat.mod_lt (pid%(4*cols)) hs
    have hle : groupSize rows cols pid≤4 := min_le_right _ _
    unfold tileRow groupBase
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by decide : 0<4)]
    rw [Nat.div_eq_of_lt (show pid%(4*cols)%groupSize rows cols pid<4 by omega)]
    simp
  have hq : a/(4*cols)=b/(4*cols) := by rw [←group_of_row a ha,←group_of_row b hb,hr]
  have hbase : groupBase rows cols a=groupBase rows cols b := by simp [groupBase,hq]
  have hsize : groupSize rows cols a=groupSize rows cols b := by simp [groupSize,hbase]
  have hmod : a%(4*cols)%groupSize rows cols a=b%(4*cols)%groupSize rows cols a := by
    unfold tileRow at hr
    rw [←hbase,←hsize] at hr
    exact Nat.add_left_cancel hr
  unfold tileCol at hc
  rw [←hsize] at hc
  exact Nat.ext_div_mod hq (Nat.ext_div_mod hc hmod)

def coordinate (rows cols : Nat) (p : Fin (rows*cols)) : Fin rows × Fin cols :=
  (⟨tileRow rows cols p.val,(grouped_bounds rows cols p.val p.isLt).2.1⟩,
   ⟨tileCol rows cols p.val,(grouped_bounds rows cols p.val p.isLt).2.2⟩)

theorem coordinate_bijective (rows cols : Nat) : Function.Bijective (coordinate rows cols) := by
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  refine ⟨?_,by simp⟩
  intro a b he
  apply Fin.ext
  exact grouped_injective rows cols a.val b.val a.isLt b.isLt
    (congrArg (fun x => x.1.val) he) (congrArg (fun x => x.2.val) he)

theorem grouped_covered (rows cols i j : Nat) (hi : i<rows) (hj : j<cols) :
    ∃ pid, pid<rows*cols ∧ tileRow rows cols pid=i ∧ tileCol rows cols pid=j := by
  obtain ⟨p,hp⟩ := (coordinate_bijective rows cols).2 (⟨i,hi⟩,⟨j,hj⟩)
  exact ⟨p.val,p.isLt,congrArg (fun x => x.1.val) hp,congrArg (fun x => x.2.val) hp⟩

@[simp] theorem source_pidM {α : Type} (r : Request) (pid : Nat) (s : State α) :
    FPValue.pidM r.parameters (instanceState r pid s)=tileRow (tileRows r) (tileColumns r) pid := by
  simp [FPValue.pidM,FPValue.firstM,FPValue.perGroup,FPValue.blocksN,FPValue.blocksM,
    FPValue.groupSize,FPValue.BM,FPValue.BN,Request.parameters,instanceState,
    tileRow,groupBase,groupSize,tileRows,tileColumns]

@[simp] theorem source_pidN {α : Type} (r : Request) (pid : Nat) (s : State α) :
    FPValue.pidN r.parameters (instanceState r pid s)=tileCol (tileRows r) (tileColumns r) pid := by
  simp [FPValue.pidN,FPValue.firstM,FPValue.perGroup,FPValue.blocksN,FPValue.blocksM,
    FPValue.groupSize,FPValue.BM,FPValue.BN,Request.parameters,instanceState,
    tileCol,groupBase,groupSize,tileRows,tileColumns]

theorem offset_injective (r : Request) {i j i' j' : Nat}
    (hj : j<r.columns) (hj' : j'<r.columns)
    (h : outputOffset r i j=outputOffset r i' j') : i=i' ∧ j=j' := by
  have hn : 0<max r.columns 1 := by omega
  have hjn : j<max r.columns 1 := lt_of_lt_of_le hj (le_max_left _ _)
  have hjn' : j'<max r.columns 1 := lt_of_lt_of_le hj' (le_max_left _ _)
  unfold outputOffset at h
  have hm := congrArg (fun n => n%max r.columns 1) h
  have hd := congrArg (fun n => n/max r.columns 1) h
  simp [Nat.add_mod,Nat.mod_eq_of_lt hjn,Nat.mod_eq_of_lt hjn'] at hm
  subst j'
  constructor
  · exact Nat.eq_of_mul_eq_mul_left hn (Nat.add_right_cancel h)
  · rfl

theorem active_injective {α : Type} (r : Request) (s : State α)
    (i j : FPValue.CIndex r.parameters)
    (hi : FPValue.active r.parameters s i) (hj : FPValue.active r.parameters s j)
    (he : FPValue.cOffset r.parameters s i=FPValue.cOffset r.parameters s j) : i=j := by
  have ho : outputOffset r (FPValue.row r.parameters s i.1) (FPValue.col r.parameters s i.2.1)=
      outputOffset r (FPValue.row r.parameters s j.1) (FPValue.col r.parameters s j.2.1) := by
    simpa [outputOffset,FPValue.cOffset,Request.parameters] using he
  have hh := offset_injective r hi.2 hj.2 ho
  have h1 : i.1=j.1 := Fin.ext (Nat.add_left_cancel hh.1)
  have h2 : i.2.1=j.2.1 := Fin.ext (Nat.add_left_cancel hh.2)
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  cases h1
  cases h2
  rfl

theorem cell_covered {α : Type} (r : Request) (s : State α)
    (row col : Nat) (hr : row<r.rows) (hc : col<r.columns) :
    ∃ pid, pid<programs r ∧ ∃ i : FPValue.CIndex r.parameters,
      FPValue.row r.parameters (instanceState r pid s) i.1=row ∧
      FPValue.col r.parameters (instanceState r pid s) i.2.1=col := by
  have hrb : row/32<tileRows r := by unfold tileRows; omega
  have hcb : col/32<tileColumns r := by unfold tileColumns; omega
  obtain ⟨pid,hp,hm,hn⟩ := grouped_covered (tileRows r) (tileColumns r) (row/32) (col/32) hrb hcb
  refine ⟨pid,hp,(⟨row%32,by change row%32<32; omega⟩,
    ⟨col%32,by change col%32<32; omega⟩,PUnit.unit),?_,?_⟩
  · change FPValue.pidM r.parameters (instanceState r pid s)*32+row%32=row
    rw [source_pidM,hm]
    omega
  · change FPValue.pidN r.parameters (instanceState r pid s)*32+col%32=col
    rw [source_pidN,hn]
    omega

theorem foldRange_pointwise {α ι : Type} (u : Nat → (ι → α) → ι → α)
    (v : Nat → α → α) (i : ι) (h : ∀ k a, u k a i=v k (a i))
    (start stop : Nat) (a : ι → α) :
    foldRange u start stop 1 a i=foldRange v start stop 1 (a i) := by
  by_cases hs : start<stop
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,hs,↓reduceDIte]
    rw [foldRange_pointwise u v i h (start+1) stop (u start a),h]
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp [hs]
termination_by stop-start
decreasing_by omega

theorem source_block {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o)
    (i : FPValue.CIndex r.parameters) (hi : FPValue.active r.parameters current i)
    (block : Nat) (acc : FPValue.CIndex r.parameters → α) :
    FPValue.update M D r.parameters .single current block acc i=
      Q.add (acc i) (blockValue Q r initial
        (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) block) := by
  have haRow : FPValue.aRow r.parameters current i.1=FPValue.row r.parameters current i.1 :=
    Nat.mod_eq_of_lt hi.1
  have hbCol : FPValue.bCol r.parameters current i.2.1=FPValue.col r.parameters current i.2.1 :=
    Nat.mod_eq_of_lt hi.2
  have ha (j : Fin 32) :
      FPValue.aValue M r.parameters .single current block (i.1,j,PUnit.unit)=
        (if block*32+j.val<r.inner then
          inputA r initial (FPValue.row r.parameters current i.1) (block*32+j.val) else Q.zero) := by
    have he : FPValue.aOffset r.parameters current block (i.1,j,PUnit.unit)=
        FPValue.row r.parameters current i.1*r.strideAM+(block*32+j.val)*r.strideAK := by
      unfold FPValue.aOffset
      rw [haRow]
      simp only [Request.parameters,FPValue.BK]
      ring
    by_cases hk : block*32+j.val<r.inner
    · have hj : j.val<r.inner-block*32 := by omega
      simp only [FPValue.aValue,Request.parameters,FPValue.BK,hj,if_true]
      change (current.mem r.A (FPValue.aOffset r.parameters current block (i.1,j,PUnit.unit))).read .fp32=_
      rw [he,hA]
      simp [inputA,hk]
      rfl
    · have hj : ¬j.val<r.inner-block*32 := by omega
      simp only [FPValue.aValue,Request.parameters,FPValue.BK,hj,if_false]
      simpa [hk,FPValue.dtype] using law.loadZero
  have hb (j : Fin 32) :
      FPValue.bValue M r.parameters .single current block (j,i.2.1,PUnit.unit)=
        (if block*32+j.val<r.inner then
          inputB r initial (block*32+j.val) (FPValue.col r.parameters current i.2.1) else Q.zero) := by
    have he : FPValue.bOffset r.parameters current block (j,i.2.1,PUnit.unit)=
        (block*32+j.val)*r.strideBK+FPValue.col r.parameters current i.2.1*r.strideBN := by
      unfold FPValue.bOffset
      rw [hbCol]
      simp only [Request.parameters,FPValue.BK]
      ring
    by_cases hk : block*32+j.val<r.inner
    · have hj : j.val<r.inner-block*32 := by omega
      simp only [FPValue.bValue,Request.parameters,FPValue.BK,hj,if_true]
      change (current.mem r.B (FPValue.bOffset r.parameters current block (j,i.2.1,PUnit.unit))).read .fp32=_
      rw [he,hB]
      simp [inputB,hk]
      rfl
    · have hj : ¬j.val<r.inner-block*32 := by omega
      simp only [FPValue.bValue,Request.parameters,FPValue.BK,hj,if_false]
      simpa [hk,FPValue.dtype] using law.loadZero
  unfold FPValue.update
  rw [law.add]
  change Q.add (acc i) (D.apply (some .fp32) .fp32 .fp32 .fp32 {}
    (batch := []) (m := 32) (k := 32) (n := 32) _ _ i)=_
  rw [law.dot]
  unfold blockValue
  apply congrArg (Q.add (acc i))
  apply congrArg₂ Q.dot
  · funext j; exact ha j
  · funext j; exact hb j

theorem source_reference {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α)
    (hA : ∀ o, current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o, current.mem r.B o=initial.mem r.B o)
    (i : FPValue.CIndex r.parameters) (hi : FPValue.active r.parameters current i) :
    FPValue.outputValue M D r.parameters .single current i=
      reference Q r initial (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) := by
  have hacc : FPValue.accumulation M D r.parameters .single current i=
      accumulator Q r initial (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) := by
    unfold FPValue.accumulation accumulator
    rw [foldRange_pointwise _ (fun block x => Q.add x (blockValue Q r initial
      (FPValue.row r.parameters current i.1) (FPValue.col r.parameters current i.2.1) block))
      i (source_block M D Q law r initial current hA hB i hi)]
    rw [law.initialZero]
    congr 1
  unfold FPValue.outputValue reference
  rw [hacc,law.halfCast]
  congr 1
  unfold FPValue.activation Numerics.activate
  simp only [Request.parameters,law.comparison,law.slope]

def Good {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (initial current : State α) (row col : Nat) : Prop :=
  current.mem r.C (outputOffset r row col)=Cell.mk .fp16 (reference Q r initial row col)

def Covered {α : Type} (r : Request) (s : State α) (ids : List Nat) (row col : Nat) : Prop :=
  ∃ pid∈ids, ∃ i : FPValue.CIndex r.parameters,
    FPValue.row r.parameters (instanceState r pid s) i.1=row ∧
    FPValue.col r.parameters (instanceState r pid s) i.2.1=col

theorem source_offset {α : Type} (r : Request) (s : State α) (i : FPValue.CIndex r.parameters) :
    FPValue.cOffset r.parameters s i=
      outputOffset r (FPValue.row r.parameters s i.1) (FPValue.col r.parameters s i.2.1) := by
  simp [FPValue.cOffset,outputOffset,Request.parameters]

theorem program_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α) (pid : Nat)
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
  obtain ⟨next,he,_,hv,hf⟩ := FPValue.correctness M D r.parameters .single start
    (active_injective r start)
  have value (i : FPValue.CIndex r.parameters) (hi : FPValue.active r.parameters start i) :
      Good Q r initial next (FPValue.row r.parameters start i.1) (FPValue.col r.parameters start i.2.1) := by
    have h := hv i hi
    rw [source_reference M D Q law r initial start hA hB i hi,source_offset] at h
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
    (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial current : State α) (ids : List Nat)
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
    obtain ⟨next,he,ha,hb,hframe,hkeep,hvalue⟩ := program_correct M D Q law r initial current pid hCA hCB hA hB
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
    (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α) (hCA : r.C≠r.A) (hCB : r.C≠r.B) :
    ∃ final, runPrograms M D r (List.range (programs r)) initial=some final ∧ Result Q r initial final := by
  obtain ⟨final,he,_,_,hf,_,hv⟩ := programs_correct M D Q law r initial initial
    (List.range (programs r)) hCA hCB (fun _ => rfl) (fun _ => rfl)
  refine ⟨final,he,?_,hf⟩
  intro row hr col hc
  obtain ⟨pid,hp,i,hi,hj⟩ := cell_covered r initial row col hr hc
  exact hv row hr col hc ⟨pid,List.mem_range.mpr hp,i,hi,hj⟩

theorem host_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α) (hCA : r.C≠r.A) (hCB : r.C≠r.B)
    (hdims : r.inner=r.innerB) (hcontiguous : r.aContiguous=Bool.true) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧ Result Q r initial final := by
  obtain ⟨final,he,hp⟩ := grid_correct M D Q law r initial hCA hCB
  exact ⟨final,by simp [host,hdims,hcontiguous,he],hp⟩

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
  have hr : tileRow (tileRows r) (tileColumns r) p*32+i.1.val=
      tileRow (tileRows r) (tileColumns r) q*32+j.1.val := by
    simpa only [FPValue.row,Request.parameters,FPValue.BM,source_pidM] using hh.1
  have hc : tileCol (tileRows r) (tileColumns r) p*32+i.2.1.val=
      tileCol (tileRows r) (tileColumns r) q*32+j.2.1.val := by
    simpa only [FPValue.col,Request.parameters,FPValue.BN,source_pidN] using hh.2
  have hpq : p=q := grouped_injective (tileRows r) (tileColumns r) p q hp hq
    (by have hi' : i.1.val<32 := i.1.isLt; have hj' : j.1.val<32 := j.1.isLt; omega)
    (by have hi' : i.2.1.val<32 := i.2.1.isLt; have hj' : j.2.1.val<32 := j.2.1.isLt; omega)
  subst q
  exact ⟨rfl,active_injective r (instanceState r p s) i j hi hj he⟩

theorem host_dimension_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (r : Request) (s : State α)
    (h : r.inner≠r.innerB) : host M D r s=.error .incompatibleDimensions := by
  simp [host,h]

theorem host_contiguity_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (r : Request) (s : State α)
    (hdims : r.inner=r.innerB) (h : r.aContiguous=Bool.false) :
    host M D r s=.error .noncontiguousA := by
  simp [host,hdims,h]

theorem host_empty {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (r : Request) (s : State α)
    (hdims : r.inner=r.innerB) (hcontiguous : r.aContiguous=Bool.true)
    (hempty : r.rows=0 ∨ r.columns=0) : host M D r s=.ok (r.C,s) := by
  rcases hempty with h | h
  all_goals simp [host,hdims,hcontiguous,programs,tileRows,tileColumns,h,runPrograms]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
