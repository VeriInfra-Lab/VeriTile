import bench.tritonbench_g.max_reduction.Proofs.Grid
import bench.tritonbench_g.max_reduction.Proofs.Native

namespace VeriTile.Bench.Repairs.MaxReduction.Machine
open VeriTile.Triton
noncomputable section
open Classical
set_option maxHeartbeats 2000000

theorem signed64 (n : Nat) (bound : n<2^63) : (Address.word n).toInt=(n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem nat32_signed (n : Nat) (bound : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem stored_index_exact (index : Nat) (bound : index<2^31) :
    indexStore index=BitVec.ofNat 64 index ∧ (indexStore index).toInt=(index : Int) := by
  have hs : (indexStore index).toInt=(index : Int) := by
    rw [indexStore,BitVec.toInt_signExtend_of_le (by decide),nat32_signed index bound]
  refine ⟨?_,hs⟩
  apply BitVec.eq_of_toInt_eq
  exact hs.trans (signed64 index (by omega)).symm

theorem pair_selects (ops : NativeOps) (p q : Word × Nat) :
    nativePair ops p q=p ∨ nativePair ops p q=q := by
  dsimp only [nativePair]
  split <;> simp_all

theorem tree_selects (tree : Tree) (op : A → A → A) (values : Nat → A)
    (select : ∀ a b,op a b=a ∨ op a b=b) :
    ∃ i,i∈tree.leaves ∧ tree.eval op values=values i := by
  induction tree with
  | leaf i => exact ⟨i,by simp [Tree.leaves],rfl⟩
  | node l r il ir =>
    rcases select (l.eval op values) (r.eval op values) with h|h
    · obtain ⟨i,hi,he⟩ := il
      exact ⟨i,List.mem_append_left _ hi,h.trans he⟩
    · obtain ⟨i,hi,he⟩ := ir
      exact ⟨i,List.mem_append_right _ hi,h.trans he⟩

def encodedPair (pair : Word × Nat) : Word × BitVec 32 := (pair.1,BitVec.ofNat 32 pair.2)

def pair32 (ops : NativeOps) (left right : Word × BitVec 32) : Word × BitVec 32 :=
  let tie := ops.orderedEqual left.1 right.1 && decide (left.2.toInt<right.2.toInt)
  let gt := ops.orderedGreater left.1 right.1 || tie
  (if gt then left.1 else right.1,if gt then left.2 else right.2)

theorem pair32_exact (ops : NativeOps) (left right : Word × Nat)
    (hl : left.2<2^31) (hr : right.2<2^31) :
    pair32 ops (encodedPair left) (encodedPair right)=encodedPair (nativePair ops left right) := by
  dsimp only [pair32,encodedPair,nativePair]
  rw [nat32_signed _ hl,nat32_signed _ hr]
  simp only [Int.ofNat_lt]
  split <;> simp_all

theorem selected_index_bound (ops : NativeOps) (tree : Tree) (values : Nat → Word × Nat)
    (bound : ∀ i,i∈tree.leaves → (values i).2<2^31) :
    (tree.eval (nativePair ops) values).2<2^31 := by
  obtain ⟨i,hi,he⟩ := tree_selects tree (nativePair ops) values (pair_selects ops)
  rw [he]
  exact bound i hi

theorem tree32_exact (ops : NativeOps) (tree : Tree) (values : Nat → Word × Nat)
    (bound : ∀ i,i∈tree.leaves → (values i).2<2^31) :
    tree.eval (pair32 ops) (fun i => encodedPair (values i))=
      encodedPair (tree.eval (nativePair ops) values) := by
  induction tree with
  | leaf i => rfl
  | node left right ihl ihr =>
    have hl : ∀ i,i∈left.leaves → (values i).2<2^31 := fun i hi => bound i (List.mem_append_left _ hi)
    have hr : ∀ i,i∈right.leaves → (values i).2<2^31 := fun i hi => bound i (List.mem_append_right _ hi)
    simp only [Tree.eval]
    rw [ihl hl,ihr hr]
    exact pair32_exact ops _ _ (selected_index_bound ops left values hl) (selected_index_bound ops right values hr)

theorem segment_start (n b p : Nat) (hb : 0<b) (hp : p<Address.count n b) : p*b<n := by
  have hprod : (p+1)*b≤n+b-1 := (Nat.le_div_iff_mul_le hb).mp hp
  simp only [Nat.add_mul,Nat.one_mul] at hprod
  omega

theorem scalar_mask (n b p lane : Nat) (hn : n<2^31) (hb : 0<b) (cap : b≤65536)
    (hp : p<Address.count n b) (hl : lane<b) :
    (Address.word p*Address.word b+Address.word lane).slt (Address.word n)=decide (p*b+lane<n) := by
  have hs := segment_start n b p hb hp
  have ho : p*b+lane<2^63 := by omega
  simp only [Address.word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.slt_eq_decide]
  change decide ((Address.word (p*b+lane)).toInt < (Address.word n).toInt)=_
  rw [signed64 _ ho,signed64 n (by omega)]
  simp only [Int.ofNat_lt]

theorem dimension_mask (a : DimArguments) (h : DimContract a) (BM p k lane n : Nat)
    (choice : BM∈blockChoices) (hp : (p,k)∈Grid.programs a.M a.K BM)
    (hl : lane<BM) (hn : n<nextPower a.N) :
    (Address.word p*Address.word BM+Address.word lane).slt (Address.word a.M)=decide (p*BM+lane<a.M) ∧
    (Address.word n).slt (Address.word a.N)=decide (n<a.N) := by
  have hg := (Grid.mem_programs _ _ _ _ _).mp hp
  have hm : 0<a.M := by have := segment_start a.M BM p (block_choice BM choice).1 hg.1; omega
  have hk : 0<a.K := by omega
  have size : a.M*a.N*a.K<2^31 := by rw [dimension_numel a h]; exact h.1.2.2.2.2.2.1
  have hnpos := h.2.2.2.1
  have mb : a.M<2^31 := lt_of_le_of_lt
    ((Nat.le_mul_of_pos_right a.M hnpos).trans (Nat.le_mul_of_pos_right (a.M*a.N) hk)) size
  have nb : a.N≤32768 := h.2.2.2.2.1
  have width : nextPower a.N≤32768 := nextPower_bound _ 15 nb
  refine ⟨scalar_mask a.M BM p lane mb (block_choice BM choice).1 (by have := (block_choice BM choice).2; omega) hg.1 hl,?_⟩
  rw [BitVec.slt_eq_decide,signed64 n (by omega),signed64 a.N (by omega)]
  simp

/-- Valid Tensor allocation bounds are ordinary backing-storage laws. Native
address multiplication uses the FP32/int64 byte width before exact readback. -/
theorem active_pointer (base count width i : Nat) (hw : 0<width)
    (backed : base+width*count≤2^64) (hi : i<count) :
    (Address.word base+Address.word i*Address.word width).toNat=base+width*i := by
  have bound : base+width*i<2^64 := by nlinarith
  simp only [Address.word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  rw [Nat.mul_comm i width,Nat.mod_eq_of_lt bound]

#axiomsClean stored_index_exact
#axiomsClean tree32_exact
#axiomsClean scalar_mask
#axiomsClean dimension_mask
#axiomsClean active_pointer
end
end VeriTile.Bench.Repairs.MaxReduction.Machine
