import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Execution

namespace VeriTile.Bench.Repairs.KvCacheCopy.Native
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 100000

theorem signed32 (n : Nat) (small : n<2^31) : (i n).toInt=(n:Int) :=
  BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n:Int)) (by decide) (by omega) (by norm_num;omega)

theorem word_nonnegative (w : Word) (nonneg : 0≤w.toInt) : i w.toInt.toNat=w := by
  unfold i
  rw [←BitVec.ofInt_natCast,Int.toNat_of_nonneg nonneg,BitVec.ofInt_toInt]

theorem ptr_exact (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap)
    (r : Role) (n : Nat) (bound : n<(m r).numel) : ptr m r (i n)=address m r n := by
  have small : n<2^31 := lt_trans bound (h.extent.1 r)
  have cap := binding.capacity r
  simp only [ptr,signed32 n small,BitVec.ofInt_natCast,BitVec.ofNat_mul_ofNat,
    BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  change (base m r+n*4)%(2^64)=base m r+4*n
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

theorem source_word (p : Config) (e : Lane) : sourceIndex p e=i (inputIndex p e.tile.sequence e.tile.head (component p e)) := by
  simp only [sourceIndex,i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  simp [inputIndex,component,Nat.add_mul,Nat.mul_assoc]

theorem source_pointer (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap)
    (e : Lane) (he : ValidLane (config m b) e) :
    ptr m (inputRole e.key) (sourceIndex (config m b) e)=sourceAddress m (config m b) e := by
  rw [source_word]
  apply ptr_exact m b l t h decode heap binding
  have bound := source_bound m b l t h e he
  unfold Inside sourceAddress address at bound
  omega

theorem resolve_exact (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap)
    (s : Nat) (hs : s<(config m b).count) :
    resolve m (config m b) decode heap s=(i (blockId (config m b) l t s),i (past l s)) := by
  have sizes := numel m b l t h
  have bounds := metadata_bounds m b l t h s hs
  have positive := (h.bounds s hs).1
  have nonneg : 0≤(l s).toInt := by omega
  have pastword : l s-i 1=i (past l s) := by
    rw [←word_nonnegative (l s) nonneg]
    exact BitVec.ofNat_sub_ofNat_of_le _ _ (by decide) (by omega)
  have colword : BitVec.ofInt 32 ((i (past l s)).toInt.tdiv ((config m b).blockSize:Int))=
      i (column (config m b) l s) := by
    rw [signed32 _ bounds.1]
    simp only [Int.natCast_tdiv_eq_ediv,←Int.natCast_ediv,BitVec.ofInt_natCast,column,i]
  have lengthPtr := ptr_exact m b l t h decode heap binding .lengths s (by rw [sizes.2.2.2.2.1];exact hs)
  have tableBound := Linear.linear_bound (config m b).count (config m b).columns s _ hs bounds.2.1
  have tablePtr := ptr_exact m b l t h decode heap binding .table _ (by rw [sizes.2.2.2.2.2];exact tableBound)
  simp only [resolve,lengthPtr,binding.lengths s hs,pastword,colword]
  have offset : i s*i (config m b).columns+i (column (config m b) l s)=
      i (s*(config m b).columns+column (config m b) l s) := by
    simp [i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  rw [offset,tablePtr,binding.table s _ hs bounds.2.1]
  congr 1
  exact (word_nonnegative _ (h.bounds s hs).2.2.1).symm

theorem target_word (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (e : Lane) (he : ValidLane (config m b) e) :
    targetIndex (config m b) (i (blockId (config m b) l t e.tile.sequence)) (i (past l e.tile.sequence)) e=
      i (outputIndex (config m b) l t e.key e.tile.sequence e.tile.head (component (config m b) e)) := by
  have geometry := split_geometry m b l t h
  have bounds := metadata_bounds m b l t h e.tile.sequence he.1.1
  have position : BitVec.ofInt 32 ((i (past l e.tile.sequence)).toInt.tmod ((config m b).blockSize:Int))=
      i (slot (config m b) l e.tile.sequence) := by
    rw [signed32 _ bounds.1,←Int.ofNat_tmod]
    rfl
  have div : component (config m b) e/(config m b).width=e.tile.split := by
    simpa [component,Nat.add_comm,Nat.div_eq_of_lt he.2] using
      Nat.add_mul_div_right e.lane e.tile.split geometry.1
  have mod : component (config m b) e%(config m b).width=e.lane := by
    simp [component,Nat.add_mod,Nat.mod_eq_of_lt he.2]
  unfold targetIndex
  rw [position]
  cases key : e.key with
  | «false» =>
    simp only [Bool.false_eq_true,ite_false,i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,
      outputIndex,Bool.false_and,cacheIndex]
    congr 1
    simp only [component]
    ring
  | «true» =>
    simp only [ite_true,i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,outputIndex,Bool.true_and]
    cases b with
    | «false» =>
      have split : e.tile.split=0 := by have hv := he.1.2.2;change e.tile.split<1 at hv;omega
      simp only [config,ite_false,Bool.false_eq_true,cacheIndex] at *
      simp only [component,split,Nat.zero_mul,Nat.zero_add]
      congr 1
      ring
    | «true» =>
      change BitVec.ofNat 32 _=BitVec.ofNat 32 _
      have new : (config m Bool.true).newLayout=Bool.true := rfl
      simp only [new,ite_true,cacheIndex,div,mod]
      congr 1
      rw [←geometry.2]
      ring

theorem target_pointer (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap)
    (e : Lane) (he : ValidLane (config m b) e) :
    ptr m (outputRole e.key) (targetIndex (config m b) (i (blockId (config m b) l t e.tile.sequence)) (i (past l e.tile.sequence)) e)=
      targetAddress m (config m b) l t e := by
  rw [target_word m b l t h e he]
  apply ptr_exact m b l t h decode heap binding
  have bound := target_bound m b l t h e he
  unfold Inside targetAddress address at bound
  omega

#axiomsClean resolve_exact
#axiomsClean source_pointer
#axiomsClean target_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy.Native
