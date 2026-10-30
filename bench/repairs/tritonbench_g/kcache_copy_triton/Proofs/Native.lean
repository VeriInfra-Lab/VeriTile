import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Execution
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.Native
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 100000

theorem signed64 (x : Nat) (small : x<2^63) : (i x).toInt=(x:Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(x:Int)) (by decide) (by omega) (by norm_num;omega)

theorem ptr_exact (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m n b l t decode heap)
    (r : Role) (x : Nat) (bound : x<(m r).numel) : ptr m r (i x)=address m r x := by
  have cap := binding.capacity r
  simp only [ptr,i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  change (base m r+x*4)%(2^64)=base m r+4*x
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

theorem source_word (p : Config) (e : Lane) : sourceIndex p e=i (inputIndex p e.tile.sequence e.tile.head (component p e)) := by
  simp only [sourceIndex,i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  simp [inputIndex,component,Nat.add_mul,Nat.mul_assoc]

theorem source_pointer (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m n b l t decode heap)
    (e : Lane) (he : ValidLane (config m n b) e) :
    ptr m .key (sourceIndex (config m n b) e)=sourceAddress m (config m n b) e := by
  rw [source_word]
  apply ptr_exact m n b l t h decode heap binding
  have bound := source_bound m n b l t h e he
  unfold Inside sourceAddress address at bound
  omega

theorem past_word (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (s : Nat) (hs : s<(config m n b).count) :
    BitVec.ofInt 64 (l (s/n)).toInt+(i s-i n*(i (s/n)+i 1))=i (past (config m n b) l s) := by
  have bounds := h.bounds.1 _ (sequence_bound m n b l t h s hs)
  change (n:Int)≤(l (s/n)).toInt ∧ _ at bounds
  have nonneg : 0≤(l (s/n)).toInt := by omega
  have leN : n≤(l (s/n)).toInt.toNat := by omega
  have partition : s/n*n+s%n=s := by simpa only [Nat.mul_comm] using Nat.div_add_mod s n
  have splitInt : (s:Int)=((s/n:Nat):Int)*(n:Int)+((s%n:Nat):Int) := by
    exact_mod_cast partition.symm
  have arithmetic : (l (s/n)).toInt+((s:Int)-(n:Int)*(((s/n:Nat):Int)+1))=(past (config m n b) l s:Int) := by
    change _=(((l (s/n)).toInt.toNat-n+s%n:Nat):Int)
    rw [Int.natCast_add,Int.ofNat_sub leN,Int.toNat_of_nonneg nonneg,splitInt]
    ring
  simp only [i,←BitVec.ofInt_natCast,←BitVec.ofInt_add,BitVec.sub_eq_add_neg,←BitVec.ofInt_neg,←BitVec.ofInt_mul]
  simpa only [Int.sub_eq_add_neg] using congrArg (BitVec.ofInt 64) arithmetic

theorem resolve_exact (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m n b l t decode heap)
    (s : Nat) (hs : s<(config m n b).count) :
    resolve m (config m n b) decode heap s=(i (blockId (config m n b) l t s),i (past (config m n b) l s)) := by
  have sizes := numel m n b l t h
  have bounds := metadata_bounds m n b l t h s hs
  have seq := sequence_bound m n b l t h s hs
  have lengthPtr := ptr_exact m n b l t h decode heap binding .lengths (s/n) (by rw [sizes.2.2.1];exact seq)
  have tableBound := Linear.linear_bound _ _ _ _ seq bounds.2.1
  have tablePtr := ptr_exact m n b l t h decode heap binding .table _ (by rw [sizes.2.2.2];exact tableBound)
  have colword : BitVec.ofInt 64 ((i (past (config m n b) l s)).toInt.tdiv ((config m n b).blockSize:Int))=
      i (column (config m n b) l s) := by
    rw [signed64 _ (by omega)]
    simp only [Int.natCast_tdiv_eq_ediv,←Int.natCast_ediv,BitVec.ofInt_natCast,column,i]
  have n_eq : (config m n b).n=n := rfl
  simp only [resolve,n_eq,lengthPtr,binding.lengths (s/n) seq,past_word m n b l t h s hs,colword]
  have offset : i (s/n)*i (config m n b).columns+i (column (config m n b) l s)=
      i (s/n*(config m n b).columns+column (config m n b) l s) := by
    simp [i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  rw [offset,tablePtr,binding.table (s/n) _ seq bounds.2.1]
  congr 1
  unfold blockId i
  rw [←BitVec.ofInt_natCast,Int.toNat_of_nonneg (h.bounds.2 s hs).1]
  rfl

theorem target_word (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e : Lane) (he : ValidLane (config m n b) e) :
    targetIndex (config m n b) (i (blockId (config m n b) l t e.tile.sequence)) (i (past (config m n b) l e.tile.sequence)) e=
      i (outputIndex (config m n b) l t e.tile.sequence e.tile.head (component (config m n b) e)) := by
  have geometry := split_geometry m n b l t h
  have bounds := metadata_bounds m n b l t h e.tile.sequence he.1.1
  have position : BitVec.ofInt 64 ((i (past (config m n b) l e.tile.sequence)).toInt.tmod ((config m n b).blockSize:Int))=
      i (slot (config m n b) l e.tile.sequence) := by
    rw [signed64 _ (by omega),←Int.ofNat_tmod]
    rfl
  have div : component (config m n b) e/(config m n b).width=e.tile.split := by
    simpa [component,Nat.add_comm,Nat.div_eq_of_lt he.2] using Nat.add_mul_div_right e.lane e.tile.split geometry.1
  have mod : component (config m n b) e%(config m n b).width=e.lane := by
    simp [component,Nat.add_mod,Nat.mod_eq_of_lt he.2]
  unfold targetIndex
  rw [position]
  simp only [i,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,outputIndex]
  cases b with
  | «false» =>
    have split : e.tile.split=0 := by have hv := he.1.2.2;change e.tile.split<1 at hv;omega
    simp only [config,ite_false,Bool.false_eq_true,KvCacheCopy.cacheIndex] at *
    simp only [component,split,Nat.zero_mul,Nat.zero_add]
    congr 1
    ring
  | «true» =>
    change BitVec.ofNat 64 _=BitVec.ofNat 64 _
    have new : (config m n Bool.true).newLayout=Bool.true := rfl
    simp only [new,ite_true,KvCacheCopy.cacheIndex,div,mod]
    congr 1
    rw [←geometry.2]
    ring

theorem target_pointer (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m n b l t decode heap)
    (e : Lane) (he : ValidLane (config m n b) e) :
    ptr m .cache (targetIndex (config m n b) (i (blockId (config m n b) l t e.tile.sequence)) (i (past (config m n b) l e.tile.sequence)) e)=
      targetAddress m (config m n b) l t e := by
  rw [target_word m n b l t h e he]
  apply ptr_exact m n b l t h decode heap binding
  have bound := target_bound m n b l t h e he
  unfold Inside targetAddress address at bound
  omega

/-- The launch IDs and arange lanes originate in signed int32. Every actual
source widening to int64 preserves the checked nonnegative coordinate. -/
theorem controls (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e : Lane) (he : ValidLane (config m n b) e) :
    (BitVec.ofNat 32 e.tile.sequence).toInt=(e.tile.sequence:Int) ∧
    (BitVec.ofNat 32 e.tile.head).toInt=(e.tile.head:Int) ∧
    (BitVec.ofNat 32 e.tile.split).toInt=(e.tile.split:Int) ∧
    (BitVec.ofNat 32 e.lane).toInt=(e.lane:Int) := by
  have grid := h.grid
  have tile := h.tile
  have exact32 : ∀ x : Nat,x<2^31 → (BitVec.ofNat 32 x).toInt=(x:Int) := by
    intro x hx
    exact BitVec.toInt_ofInt_eq_self (w:=32) (n:=(x:Int)) (by decide) (by omega) (by norm_num;omega)
  have hs := he.1.1
  have hh := he.1.2.1
  have hq := he.1.2.2
  have hc := he.2
  exact ⟨exact32 _ (by omega),exact32 _ (by omega),exact32 _ (by omega),exact32 _ (by omega)⟩

#axiomsClean controls
#axiomsClean past_word
#axiomsClean resolve_exact
#axiomsClean source_pointer
#axiomsClean target_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.Native
