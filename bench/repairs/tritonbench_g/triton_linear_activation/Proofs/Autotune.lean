import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.HostReturn
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

theorem run_absorbs (a : Arithmetic α) (s : Shape) (b : Buffers) (c trial : Fin 98)
    (mode : Mode) (name : Option String) (sep : Separate s b) (entry : Memory α) :
    run a s b c mode name (run a s b trial mode name entry)=run a s b c mode name entry := by
  have reads (p : Address) (read : b.reads s p) :
      run a s b trial mode name entry p=entry p :=
    run_frame a s b trial mode name sep entry p (fun write => sep.1 p read write.1)
  funext p
  by_cases active : b.writesStorage p ∧ p.2<s.m*s.n
  · have hn : 0<s.n := by have := active.2; nlinarith
    have maxn : max s.n 1=s.n := max_eq_left hn
    have hm : p.2/max s.n 1<s.m := by rw [maxn]; exact (Nat.div_lt_iff_lt_mul hn).mpr active.2
    have hc : p.2%max s.n 1<s.n := by rw [maxn]; exact Nat.mod_lt _ hn
    have off : outputOffset s (p.2/max s.n 1) (p.2%max s.n 1)=p.2 := by
      unfold outputOffset
      simpa only [Nat.mul_comm] using Nat.div_add_mod p.2 (max s.n 1)
    have addr : (p.1,outputOffset s (p.2/max s.n 1) (p.2%max s.n 1))=p := by rw [off]
    rw [←addr,run_cell a s b c mode name sep _ _ _ _ hm hc active.1,
      run_cell a s b c mode name sep _ _ _ _ hm hc active.1,
      preactivation_reads a s b c mode entry _ reads]
  · rw [run_frame a s b c mode name sep _ p active,
      run_frame a s b trial mode name sep entry p active,
      run_frame a s b c mode name sep entry p active]

def trialRun (a : Arithmetic α) (s : Shape) (b : Buffers) (mode : Mode) (name : Option String)
    (trials : List (Fin 98)) (entry : Memory α) : Memory α :=
  Schedule.run (fun c => run a s b c mode name) trials entry

theorem trial_absorbed (a : Arithmetic α) (s : Shape) (b : Buffers) (mode : Mode) (name : Option String)
    (trials : List (Fin 98)) (c : Fin 98) (sep : Separate s b) (entry : Memory α) :
    run a s b c mode name (trialRun a s b mode name trials entry)=run a s b c mode name entry := by
  induction trials generalizing entry with
  | nil => rfl
  | cons first tail ih =>
    change run a s b c mode name (trialRun a s b mode name tail (run a s b first mode name entry))=_
    rw [ih,run_absorbs a s b c first mode name sep entry]

variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}

theorem Allocation.eval_autotunedKernelCode (h : Allocation r caller before) (a : Arithmetic α)
    (trials : List (Fin 98)) (c : Fin 98) (valid : r.Valid) :
    (h.autotunedKernelCode trials c).eval a (h.prepared a)=((),h.kernel a c) := by
  unfold Allocation.autotunedKernelCode Allocation.kernel
  split_ifs with launch
  · have sep := h.separate valid launch.1 launch.2
    simp only [bind,eval_bind]
    rw [eval_sequenceItems a _ _ (fun q m => eval_gridCode a _ _ q _ _ m sep)]
    simp only [Prod.fst,Prod.snd]
    rw [eval_gridCode a _ _ c _ _ _ sep]
    exact congrArg (fun m => ((),m)) (trial_absorbed a _ _ _ _ trials c sep (h.prepared a))
  · rfl

theorem Allocation.eval_autotunedCode (h : Allocation r caller before) (a : Arithmetic α)
    (trials : List (Fin 98)) (c : Fin 98) (valid : r.Valid) :
    (h.autotunedCode trials c).eval a h.initial=((),h.finalMemory a c) := by
  simp only [Allocation.autotunedCode,bind,eval_bind,h.eval_prepareCode a,Prod.fst,Prod.snd,
    h.eval_autotunedKernelCode a trials c valid,h.eval_copybackCode a,Allocation.finalMemory]

theorem Allocation.native_autotuned (h : Allocation r caller before) (a : Arithmetic α)
    (trials : List (Fin 98)) (c : Fin 98) (valid : r.Valid) (abi : LocalEventABI α σ a)
    (state : σ) (entry : abi.observe state=h.initial) :
    abi.observe ((h.autotunedCode trials c).native abi state).2=h.finalMemory a c := by
  have proof := Code.native_refines abi (h.autotunedCode trials c) state
  rw [entry,h.eval_autotunedCode a trials c valid] at proof
  exact congrArg Prod.snd proof

#axiomsClean run_absorbs
#axiomsClean trial_absorbed
#axiomsClean Allocation.eval_autotunedCode
#axiomsClean Allocation.native_autotuned
end
end VeriTile.Bench.Repairs.TritonLinearActivation

