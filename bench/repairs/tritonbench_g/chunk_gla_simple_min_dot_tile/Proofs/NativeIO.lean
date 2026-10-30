import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.Native
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
open Native (word)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def row (c : Shape) (q : FPGrid.Pid c) (i : Fin c.BT) : BitVec 64 :=
  word (launchIndex c q).chunk * word c.BT + word i.val
def column (c : Shape) (q : FPGrid.Pid c) (i : Fin c.BV) : BitVec 64 :=
  word (launchIndex c q).value * word c.BV + word i.val
def key (c : Shape) (block : Nat) (i : Fin c.BK) : BitVec 64 := word block*word c.BK+word i.val

def qOffset (c : Shape) (q : FPGrid.Pid c) (block : Nat) (i : FPValue.QIndex c) :=
  word (launchIndex c q).head*(word c.T*word c.K)+row c q i.1*word c.K+key c block i.2.1
def kOffset (c : Shape) (q : FPGrid.Pid c) (block : Nat) (i : FPValue.KIndex c) :=
  word (launchIndex c q).head*(word c.T*word c.K)+key c block i.1+row c q i.2.1*word c.K
def hOffset (c : Shape) (q : FPGrid.Pid c) (block : Nat) (i : FPValue.HIndex c) :=
  word (launchIndex c q).head*(word c.NT*word c.K*word c.V)+
    word (launchIndex c q).chunk*word c.K*word c.V+key c block i.1*word c.V+column c q i.2.1
def oOffset (c : Shape) (q : FPGrid.Pid c) (i : FPValue.OIndex c) :=
  word (launchIndex c q).head*(word c.T*word c.V)+row c q i.1*word c.V+column c q i.2.1
def gOffset (c : Shape) (q : FPGrid.Pid c) (i : Fin c.BT) :=
  word (launchIndex c q).head*word c.T+row c q i

def active (c : Shape) (q : FPGrid.Pid c) (i : FPValue.OIndex c) : Prop :=
  (row c q i.1).toInt<(word c.T).toInt ∧ (column c q i.2.1).toInt<(word c.V).toInt
instance (c : Shape) (q : FPGrid.Pid c) : DecidablePred (active c q) := fun _ => inferInstanceAs (Decidable (_∧_))

def qValue (M : Algebra α) (c : Shape) (s : State α) (q : FPGrid.Pid c)
    (block : Nat) (i : FPValue.QIndex c) :=
  M.fp32Load (if (row c q i.1).toInt<(word c.T).toInt ∧ (key c block i.2.1).toInt<(word c.K).toInt
    then (s.mem "q" (qOffset c q block i).toNat).read .real else FPValue.zero M)
def kValue (M : Algebra α) (c : Shape) (s : State α) (q : FPGrid.Pid c)
    (block : Nat) (i : FPValue.KIndex c) :=
  M.fp32Load (if (key c block i.1).toInt<(word c.K).toInt ∧ (row c q i.2.1).toInt<(word c.T).toInt
    then (s.mem "k" (kOffset c q block i).toNat).read .real else FPValue.zero M)
def hValue (M : Algebra α) (c : Shape) (s : State α) (q : FPGrid.Pid c)
    (block : Nat) (i : FPValue.HIndex c) :=
  M.fp32Load (if (key c block i.1).toInt<(word c.K).toInt ∧ (column c q i.2.1).toInt<(word c.V).toInt
    then (s.mem "h" (hOffset c q block i).toNat).read .real else FPValue.zero M)
def vValue (M : Algebra α) (c : Shape) (s : State α) (q : FPGrid.Pid c) (i : FPValue.OIndex c) :=
  M.fp32Load (if active c q i then (s.mem "v" (oOffset c q i).toNat).read .real else FPValue.zero M)
def gValue (M : Algebra α) (c : Shape) (s : State α) (q : FPGrid.Pid c) (i : Fin c.BT) :=
  M.fp32Load (if (row c q i).toInt<(word c.T).toInt then (s.mem "g" (gOffset c q i).toNat).read .real else FPValue.zero M)

theorem nat_of_int (w : BitVec 64) (n : Nat) (h : w.toInt=(n : Int)) : w.toNat=n := by
  have hz : 0≤w.toInt := by omega
  rw [BitVec.toInt_eq_toNat_of_lt (BitVec.toInt_pos_iff.mp hz)] at h
  exact Int.ofNat_inj.mp h

theorem pids (c : Shape) (s : State α) (q : FPGrid.Pid c) :
    (FPGrid.launch c s q).pids 0=(launchIndex c q).value ∧
    (FPGrid.launch c s q).pids 1=(launchIndex c q).chunk ∧
    (FPGrid.launch c s q).pids 2=(launchIndex c q).head := by
  simp [FPGrid.launch,HostFP.atProgram,GridIndex.toPids,launchIndex,grid,Shape.grid,Grid.rank]

theorem masks (c : Shape) (hf : Native.Fits c) (hn : 0<c.heads*c.T*c.V)
    (hb : c.BT<2^32) (q : FPGrid.Pid c) :
    (∀ i, (row c q i).toInt<(word c.T).toInt ↔ (launchIndex c q).chunk*c.BT+i.val<c.T) ∧
    (∀ i, (column c q i).toInt<(word c.V).toInt ↔ (launchIndex c q).value*c.BV+i.val<c.V) ∧
    (∀ block, block<(c.K+c.BK-1)/c.BK → ∀ i,
      (key c block i).toInt<(word c.K).toInt ↔ block*c.BK+i.val<c.K) := by
  obtain ⟨ht,hk,hv⟩ := Native.dimensions_fit c hf hn
  obtain ⟨hqv,hqt,_⟩ := launch_lt c q
  have hbk : c.BK<2^32 := lt_of_le_of_lt c.bk_max (by decide)
  have hbv : c.BV<2^32 := lt_of_le_of_lt (Nat.min_le_right _ _) (by decide)
  exact ⟨fun i => Native.mask_exact _ _ _ _ ht hb hqt i.isLt,
    fun i => Native.mask_exact _ _ _ _ hv hbv hqv i.isLt,
    fun block hk' i => Native.mask_exact _ _ _ _ hk hbk hk' i.isLt⟩

theorem loads (M : Algebra α) (c : Shape) (hf : Native.Fits c) (hn : 0<c.heads*c.T*c.V)
    (hb : c.BT<2^32) (s : State α) (q : FPGrid.Pid c) (block : Nat)
    (hk : block<(c.K+c.BK-1)/c.BK) :
    qValue M c s q block=FPValue.qValue M c (FPGrid.launch c s q) block ∧
    kValue M c s q block=FPValue.kValue M c (FPGrid.launch c s q) block ∧
    hValue M c s q block=FPValue.hValue M c (FPGrid.launch c s q) block := by
  obtain ⟨mr,mc,mk⟩ := masks c hf hn hb q
  obtain ⟨_,hqt,hqh⟩ := launch_lt c q
  have hp := pids c s q
  refine ⟨?_,?_,?_⟩
  · funext i
    simp only [qValue,FPValue.qValue,FPValue.row,FPValue.key,hp.2.1,mr,mk block hk]
    split_ifs with ha
    · have he := nat_of_int _ _ (Native.query_exact c hf (launchIndex c q) hqh _ _ ha.1 ha.2)
      simp only [qOffset,row,key,←BitVec.ofNat_mul,←BitVec.ofNat_add] at he ⊢
      simpa only [FPValue.qOffset,FPValue.row,FPValue.key,hp.2.1,hp.2.2,FPGrid.launch,HostFP.atProgram] using congrArg (fun n => M.fp32Load ((s.mem "q" n).read .real)) he
    · rfl
  · funext i
    simp only [kValue,FPValue.kValue,FPValue.row,FPValue.key,hp.2.1,mr,mk block hk]
    split_ifs with ha
    · have he := nat_of_int _ _ (Native.key_exact c hf (launchIndex c q) hqh _ _ ha.2 ha.1)
      simp only [kOffset,row,key,←BitVec.ofNat_mul,←BitVec.ofNat_add] at he ⊢
      simpa only [FPValue.kOffset,FPValue.row,FPValue.key,hp.2.1,hp.2.2,FPGrid.launch,HostFP.atProgram] using congrArg (fun n => M.fp32Load ((s.mem "k" n).read .real)) he
    · rfl
  · funext i
    simp only [hValue,FPValue.hValue,FPValue.column,FPValue.key,hp.1,mc,mk block hk]
    split_ifs with ha
    · have he := nat_of_int _ _ (Native.history_exact c hf (launchIndex c q) hqh hqt _ _ ha.1 ha.2)
      simp only [hOffset,column,key,←BitVec.ofNat_mul,←BitVec.ofNat_add] at he ⊢
      simpa only [FPValue.hOffset,FPValue.column,FPValue.key,hp.1,hp.2.1,hp.2.2,FPGrid.launch,HostFP.atProgram] using congrArg (fun n => M.fp32Load ((s.mem "h" n).read .real)) he
    · rfl

theorem active_eq (c : Shape) (hf : Native.Fits c) (hn : 0<c.heads*c.T*c.V)
    (hb : c.BT<2^32) (s : State α) (q : FPGrid.Pid c) (i : FPValue.OIndex c) :
    active c q i ↔ FPValue.active c (FPGrid.launch c s q) i := by
  have hm := masks c hf hn hb q
  have hp := pids c s q
  simp only [active,FPValue.active,FPValue.row,FPValue.column,hm.1,hm.2.1,hp.1,hp.2.1]

theorem output_offset (c : Shape) (hf : Native.Fits c) (hn : 0<c.heads*c.T*c.V)
    (hb : c.BT<2^32) (s : State α) (q : FPGrid.Pid c) (i : FPValue.OIndex c)
    (hi : active c q i) : (oOffset c q i).toNat=FPValue.oOffset c (FPGrid.launch c s q) i := by
  have hm := masks c hf hn hb q
  have hp := pids c s q
  obtain ⟨hr,hv⟩ := hi
  have he := nat_of_int _ _ (Native.value_output_exact c hf (launchIndex c q) (launch_lt c q).2.2
    _ _ ((hm.1 _).mp hr) ((hm.2.1 _).mp hv))
  simpa only [oOffset,row,column,FPValue.oOffset,FPValue.row,FPValue.column,hp.1,hp.2.1,hp.2.2,
    ←BitVec.ofNat_mul,←BitVec.ofNat_add] using he

theorem tail_loads (M : Algebra α) (c : Shape) (hf : Native.Fits c) (hn : 0<c.heads*c.T*c.V)
    (hb : c.BT<2^32) (s : State α) (q : FPGrid.Pid c) :
    vValue M c s q=FPValue.vValue M c (FPGrid.launch c s q) ∧
    gValue M c s q=FPValue.gValue M c (FPGrid.launch c s q) := by
  have hm := masks c hf hn hb q
  have hp := pids c s q
  constructor
  · funext i
    simp only [vValue,FPValue.vValue,←active_eq c hf hn hb s q i]
    split_ifs with hi
    · rw [output_offset c hf hn hb s q i hi]
      rfl
    · rfl
  · funext i
    simp only [gValue,FPValue.gValue,FPValue.row,hp.2.1,hm.1]
    split_ifs with hi
    · have he := nat_of_int _ _ (Native.gate_exact c hf (launchIndex c q) (launch_lt c q).2.2 _ hi)
      simp only [gOffset,row,←BitVec.ofNat_mul,←BitVec.ofNat_add] at he ⊢
      simpa only [FPValue.gOffset,FPValue.row,hp.2.1,hp.2.2,FPGrid.launch,HostFP.atProgram] using
        congrArg (fun n => M.fp32Load ((s.mem "g" n).read .real)) he
    · rfl

def update (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (s : State α) (q : FPGrid.Pid c)
    (block : Nat) (a : FPValue.Acc α c) : FPValue.Acc α c :=
  (fun i => M.binary (some .fp32) .real .add (a.1 i) (d.qh (qValue M c s q block) (hValue M c s q block) i),
   fun i => M.binary (some .fp32) .real .add (a.2 i) (d.qk (qValue M c s q block) (kValue M c s q block) i))

theorem update_eq (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (hf : Native.Fits c)
    (hn : 0<c.heads*c.T*c.V) (hb : c.BT<2^32) (s : State α) (q : FPGrid.Pid c)
    (block : Nat) (hk : block<(c.K+c.BK-1)/c.BK) :
    update M c d s q block=FPValue.update M c d (FPGrid.launch c s q) block := by
  obtain ⟨hq,hk,hh⟩ := loads M c hf hn hb s q block hk
  funext a
  simp only [update,FPValue.update,hq,hk,hh]

#axiomsClean loads
#axiomsClean tail_loads
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeIO
