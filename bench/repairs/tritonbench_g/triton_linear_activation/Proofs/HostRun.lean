import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.HostData
import bench.repairs.tritonbench_g.triton_linear_activation.Operations
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}

def savedValue (a : Arithmetic α) (mode : Mode) (z : α) : α :=
  match mode with
  | .fp32 => a.encode .fp32 z
  | .autocastFP16 => a.encode .fp32 (a.decode .autocastFP16 (a.encode .autocastFP16 z))

theorem outputOffset_bound (s : Shape) (m n : Nat) (hm : m<s.m) (hn : n<s.n) :
    outputOffset s m n<s.m*s.n := by
  have hn0 : 0<s.n := by omega
  simp only [outputOffset,max_eq_left (show 1≤s.n by omega)]
  have := Nat.mul_le_mul_right s.n (Nat.succ_le_of_lt hm)
  nlinarith

theorem Allocation.kernel_output (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n) :
    h.kernel a c (h.storage 0,outputOffset r.shape m n)=
      a.encode r.mode (activation a r.activation (callerPre a r c before m n)) := by
  have launched : r.launch := ⟨by omega,by omega⟩
  simp only [Allocation.kernel,if_pos launched]
  have out := run_output a r.shape h.buffers c r.mode r.activation
    (h.separate valid launched.1 launched.2) (h.prepared a) m n hm hn
  rw [h.preactivation a c m n hm hn] at out
  exact out

theorem Allocation.kernel_saved (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) (m n storage : Nat) (hm : m<r.batch.rows) (hn : n<r.n)
    (saved : h.buffers.saved=some storage) :
    h.kernel a c (storage,outputOffset r.shape m n)=a.encode r.mode (callerPre a r c before m n) := by
  have launched : r.launch := ⟨by omega,by omega⟩
  simp only [Allocation.kernel,if_pos launched]
  rw [run_saved a r.shape h.buffers c r.mode r.activation
    (h.separate valid launched.1 launched.2) (h.prepared a) m n storage hm hn saved,
    h.preactivation a c m n hm hn]

theorem Allocation.kernel_frame (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) (p : Address) (outside : ¬(h.buffers.writesStorage p ∧ p.2<r.batch.rows*r.n)) :
    h.kernel a c p=h.prepared a p := by
  unfold Allocation.kernel
  split_ifs with launched
  · exact run_frame a r.shape h.buffers c r.mode r.activation
      (h.separate valid launched.1 launched.2) (h.prepared a) p outside
  · rfl

theorem Allocation.copyback_frame (h : Allocation r caller before) (a : Arithmetic α)
    (memory : Memory α) (p : Address)
    (outside : ¬(r.hasSaved=true ∧ p.1=r.storage 3 ∧ p.2<r.batch.rows*r.n)) :
    h.copyback a memory p=memory p := by
  cases mode : r.mode with
  | fp32 => simp only [Allocation.copyback,mode]
  | autocastFP16 =>
    simp only [Allocation.copyback,mode]
    split_ifs with saved
    · apply copy_frame a .autocastFP16 .fp32 _ _ _ (h.fresh_input 4 3)
      by_cases eq : p.1=r.storage 3
      · exact Or.inr (by by_contra ht; exact outside ⟨saved,eq,by omega⟩)
      · exact Or.inl eq
    · rfl

theorem Allocation.final_output (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n) :
    h.finalMemory a c (h.storage 0,outputOffset r.shape m n)=
      a.encode r.mode (activation a r.activation (callerPre a r c before m n)) := by
  rw [Allocation.finalMemory,h.copyback_frame a _ _ (by
    intro eq; exact h.fresh_input 0 3 eq.2.1)]
  exact h.kernel_output a c valid m n hm hn

theorem Allocation.final_saved (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n) (saved : r.hasSaved=true) :
    h.finalMemory a c (r.storage 3,outputOffset r.shape m n)=
      savedValue a r.mode (callerPre a r c before m n) := by
  cases mode : r.mode with
  | fp32 =>
    simp only [Allocation.finalMemory,Allocation.copyback,mode,savedValue]
    have hs : h.buffers.saved=some (r.storage 3) := by simp [Allocation.buffers,saved,mode]
    simpa only [mode] using h.kernel_saved a c valid m n _ hm hn hs
  | autocastFP16 =>
    simp only [Allocation.finalMemory,Allocation.copyback,mode,saved,ite_true,savedValue]
    rw [copy_output a .autocastFP16 .fp32 (h.storage 4) (r.storage 3) (r.batch.rows*r.n)
      (h.fresh_input 4 3) (h.kernel a c) (outputOffset r.shape m n)
      (show outputOffset r.shape m n<r.batch.rows*r.n from outputOffset_bound r.shape m n hm hn)]
    have hs : h.buffers.saved=some (h.storage 4) := by simp [Allocation.buffers,saved,mode]
    rw [h.kernel_saved a c valid m n _ hm hn hs,mode]

theorem Allocation.final_frame (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) (p : Address) (hp : p.1∈caller)
    (outside : ¬(r.hasSaved=true ∧ p.1=r.storage 3 ∧ p.2<r.batch.rows*r.n)) :
    h.finalMemory a c p=before p := by
  rw [Allocation.finalMemory,h.copyback_frame a _ p outside,h.kernel_frame a c valid p,
    h.prepared_frame a p hp]
  intro write
  have noFresh (i : Fin 6) : p.1≠h.storage i := by intro eq; exact h.fresh i (eq ▸ hp)
  rcases write.1 with output|saved
  · exact noFresh 0 output
  · by_cases flag : r.hasSaved=true
    · have se : (match r.mode with | .fp32 => r.storage 3 | .autocastFP16 => h.storage 4)=p.1 := by
        simpa [Allocation.buffers,flag] using saved
      cases mode : r.mode with
      | fp32 =>
        simp only [mode] at se
        exact outside ⟨flag,se.symm,write.2⟩
      | autocastFP16 =>
        simp only [mode] at se
        exact noFresh 4 se.symm
    · simp [Allocation.buffers,flag] at saved

def Allocation.kernelCode (h : Allocation r caller before) (c : Fin 98) : Code α Unit :=
  if r.launch then gridCode r.shape h.buffers c r.mode r.activation else pure ()

def Allocation.hostCode (h : Allocation r caller before) (c : Fin 98) : Code α Unit := do
  let _ ← h.prepareCode
  let _ ← h.kernelCode c
  h.copybackCode

theorem Allocation.eval_prepareCode (h : Allocation r caller before) (a : Arithmetic α) :
    h.prepareCode.eval a h.initial=((),h.prepared a) := by
  cases mode : r.mode <;>
    simp [Allocation.prepareCode,Allocation.prepared,mode,CopyRegions.eval_code,pure]

theorem Allocation.eval_kernelCode (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) :
    (h.kernelCode c).eval a (h.prepared a)=((),h.kernel a c) := by
  unfold Allocation.kernelCode Allocation.kernel
  split_ifs with launched
  · exact eval_gridCode a _ _ _ _ _ _ (h.separate valid launched.1 launched.2)
  · rfl

theorem Allocation.eval_copybackCode (h : Allocation r caller before) (a : Arithmetic α) (memory : Memory α) :
    h.copybackCode.eval a memory=((),h.copyback a memory) := by
  cases mode : r.mode <;> cases saved : r.hasSaved <;>
    simp [Allocation.copybackCode,Allocation.copyback,mode,saved,pure,eval_copyCode]

theorem Allocation.eval_hostCode (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) : (h.hostCode c).eval a h.initial=((),h.finalMemory a c) := by
  simp only [Allocation.hostCode,bind,eval_bind,h.eval_prepareCode a,Prod.fst,Prod.snd,
    h.eval_kernelCode a c valid,h.eval_copybackCode a,Allocation.finalMemory]

/-- Whole host memory refinement is composed from one-event laws, including
every input cast and caller saved copy. There is no whole-host simulation premise. -/
theorem Allocation.native_host (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (valid : r.Valid) (abi : LocalEventABI α σ a) (state : σ) (entry : abi.observe state=h.initial) :
    abi.observe ((h.hostCode c).native abi state).2=h.finalMemory a c := by
  have proof := Code.native_refines abi (h.hostCode c) state
  rw [entry,h.eval_hostCode a c valid] at proof
  exact congrArg Prod.snd proof

#axiomsClean Allocation.final_output
#axiomsClean Allocation.final_saved
#axiomsClean Allocation.final_frame
#axiomsClean Allocation.native_host
end
end VeriTile.Bench.Repairs.TritonLinearActivation
