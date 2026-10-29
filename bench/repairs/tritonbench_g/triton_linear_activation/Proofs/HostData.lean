import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.HostTypes
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}

theorem separation_of_regions (s : Shape) (b : Buffers)
    (ox : b.out≠b.x) (ow : b.out≠b.w)
    (ob : ∀ p,b.bias=some p → b.out≠p)
    (os : ∀ p,b.saved=some p → b.out≠p)
    (sx : ∀ p,b.saved=some p → 0<s.k → b.x≠p ∧ b.w≠p)
    (sb : ∀ p q,b.saved=some p → b.bias=some q → q≠p) : Separate s b := by
  unfold Separate Buffers.reads Buffers.writesStorage
  constructor
  · intro p read write
    rcases write with write|write
    · rcases read with ⟨hk,hx|hw⟩|hb
      · exact ox (write.symm.trans hx)
      · exact ow (write.symm.trans hw)
      · exact ob p.1 hb write.symm
    · rcases read with ⟨hk,hx|hw⟩|hb
      · exact (sx p.1 write hk).1 hx.symm
      · exact (sx p.1 write hk).2 hw.symm
      · exact sb p.1 p.1 write hb rfl
  · intro he
    exact os b.out he rfl

 theorem Allocation.storage_ne (allocation : Allocation r caller before) (i j : Fin 6) (h : i≠j) :
    allocation.storage i≠allocation.storage j := fun eq => h (allocation.distinct eq)

 theorem Allocation.separate (allocation : Allocation r caller before) (valid : r.Valid)
    (hm : 0<r.batch.rows) (hn : 0<r.n) : Separate r.shape allocation.buffers := by
  have savedX (hs : r.hasSaved=true) (hk : 0<r.k) : r.storage 0≠r.storage 3 := by
    apply valid.2 0 3 (by decide)
    · simpa [Request.count] using Nat.mul_pos hm hk
    · simpa [Request.count,hs] using Nat.mul_pos hm hn
  have savedW (hs : r.hasSaved=true) (hk : 0<r.k) : r.storage 1≠r.storage 3 := by
    apply valid.2 1 3 (by decide)
    · simpa [Request.count] using Nat.mul_pos hn hk
    · simpa [Request.count,hs] using Nat.mul_pos hm hn
  have savedBias (hs : r.hasSaved=true) (hb : r.hasBias=true) : r.storage 2≠r.storage 3 := by
    apply valid.2 2 3 (by decide)
    · simpa [Request.count,hb] using hn
    · simpa [Request.count,hs] using Nat.mul_pos hm hn
  cases mode : r.mode with
  | fp32 =>
    apply separation_of_regions
    · simpa [Allocation.buffers,Allocation.computeStorage,mode,inputSlot] using allocation.fresh_input 0 0
    · simpa [Allocation.buffers,Allocation.computeStorage,mode,inputSlot] using allocation.fresh_input 0 1
    · intro p hp
      cases hb : r.hasBias <;> simp [Allocation.buffers,Allocation.computeStorage,mode,hb,inputSlot,castSlot] at hp
      subst p; exact allocation.fresh_input 0 2
    · intro p hp
      cases hs : r.hasSaved <;> simp [Allocation.buffers,mode,hs] at hp
      subst p; exact allocation.fresh_input 0 3
    · intro p hp hk
      cases hs : r.hasSaved <;> simp [Allocation.buffers,mode,hs] at hp
      subst p
      simpa [Allocation.buffers,Allocation.computeStorage,mode,inputSlot] using And.intro (savedX hs hk) (savedW hs hk)
    · intro p q hp hq
      cases hs : r.hasSaved <;> simp [Allocation.buffers,mode,hs] at hp
      cases hb : r.hasBias <;> simp [Allocation.buffers,Allocation.computeStorage,mode,hb,inputSlot,castSlot] at hq
      subst p; subst q; exact savedBias hs hb
  | autocastFP16 =>
    apply separation_of_regions
    · simpa [Allocation.buffers,Allocation.computeStorage,mode,castSlot] using allocation.storage_ne 0 1 (by decide)
    · simpa [Allocation.buffers,Allocation.computeStorage,mode,castSlot] using allocation.storage_ne 0 2 (by decide)
    · intro p hp
      cases hb : r.hasBias <;> simp [Allocation.buffers,Allocation.computeStorage,mode,hb,inputSlot,castSlot] at hp
      subst p; exact allocation.storage_ne 0 3 (by decide)
    · intro p hp
      cases hs : r.hasSaved <;> simp [Allocation.buffers,mode,hs] at hp
      subst p; exact allocation.storage_ne 0 4 (by decide)
    · intro p hp hk
      cases hs : r.hasSaved <;> simp [Allocation.buffers,mode,hs] at hp
      subst p
      simpa [Allocation.buffers,Allocation.computeStorage,mode,castSlot] using
        And.intro (allocation.storage_ne 1 4 (by decide)) (allocation.storage_ne 2 4 (by decide))
    · intro p q hp hq
      cases hs : r.hasSaved <;> simp [Allocation.buffers,mode,hs] at hp
      cases hb : r.hasBias <;> simp [Allocation.buffers,Allocation.computeStorage,mode,hb,inputSlot,castSlot] at hq
      subst p; subst q; exact allocation.storage_ne 3 4 (by decide)

def inputConversion (a : Arithmetic α) (mode : Mode) (raw : α) : α :=
  a.decode mode (match mode with
    | .fp32 => raw
    | .autocastFP16 => a.encode .autocastFP16 (a.decode .fp32 raw))

/-- Independent caller-based numerical target. FP16 conversions, bias seed,
all padded instruction groups, default fusion and final store are explicit. -/
def callerPre (a : Arithmetic α) (r : Request) (c : Fin 98) (before : Memory α) (m n : Nat) : α :=
  accumulation a c r.mode r.k
    (fun k => inputConversion a r.mode (before (r.storage 0,m*max r.k 1+k)))
    (fun k => inputConversion a r.mode (before (r.storage 1,n*max r.k 1+k)))
    (if r.hasBias then some (inputConversion a r.mode (before (r.storage 2,n))) else none)

theorem Allocation.preactivation (allocation : Allocation r caller before) (a : Arithmetic α)
    (c : Fin 98) (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n) :
    preactivation a r.shape allocation.buffers c r.mode (allocation.prepared a) m n=callerPre a r c before m n := by
  have inputBound (rows K row k : Nat) (hr : row<rows) (hk : k<K) : row*max K 1+k<rows*K := by
    have hp : 0<K := by omega
    rw [max_eq_left hp]
    have mul := Nat.mul_le_mul_right K (Nat.succ_le_of_lt hr)
    nlinarith
  have hx (k : Nat) (hk : k<r.k) :
      a.decode r.mode (allocation.prepared a (allocation.computeStorage 0,m*max r.k 1+k))=
        inputConversion a r.mode (before (r.storage 0,m*max r.k 1+k)) := by
    rw [allocation.prepared_input a 0 _ (by simpa [Request.count,inputSlot] using inputBound _ _ _ _ hm hk)]
    rfl
  have hw (k : Nat) (hk : k<r.k) :
      a.decode r.mode (allocation.prepared a (allocation.computeStorage 1,n*max r.k 1+k))=
        inputConversion a r.mode (before (r.storage 1,n*max r.k 1+k)) := by
    rw [allocation.prepared_input a 1 _ (by simpa [Request.count,inputSlot] using inputBound _ _ _ _ hn hk)]
    rfl
  have hb (flag : r.hasBias=true) :
      a.decode r.mode (allocation.prepared a (allocation.computeStorage 2,n))=
        inputConversion a r.mode (before (r.storage 2,n)) := by
    rw [allocation.prepared_input a 2 n (by simpa [Request.count,inputSlot,flag] using hn)]
    rfl
  simp only [TritonLinearActivation.preactivation,callerPre,Allocation.buffers,Request.shape]
  cases flag : r.hasBias
  · simp only [flag,Bool.false_eq_true,ite_false,Option.map_none]
    exact accumulation_inputs a c r.mode r.k _ _ _ _ none hx hw
  · simp only [flag,ite_true,TritonLinearActivation.preactivation,Option.map_some,hb flag]
    exact accumulation_inputs a c r.mode r.k _ _ _ _ _ hx hw

#axiomsClean Allocation.separate
#axiomsClean Allocation.preactivation
end
end VeriTile.Bench.Repairs.TritonLinearActivation
