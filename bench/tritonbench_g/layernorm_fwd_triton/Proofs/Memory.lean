import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Address
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Mathematical_FPContract

/-! Local four-byte memory semantics. The caller frame observes every byte,
including bytes outside tensor views. No whole execution or output relation is
an API field. The Real replay instantiates these local operations independently
of native payload decoding. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Memory
noncomputable section
open scoped Classical
open VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages (fold_preserves)

/-- Non-overlap of complete FP32 windows, with arbitrary byte bases. -/
def Apart (a b : Nat) : Prop := a+4 ≤ b  ∨  b+4 ≤ a

theorem Apart.symm {a b : Nat} (h : Apart a b) : Apart b a := Or.symm h

theorem Apart.ne {a b : Nat} (h : Apart a b) : a ≠ b := by
  intro he
  subst b
  unfold Apart at h
  omega

structure API (α Store Byte : Type) where
  read : Store → Nat → α
  write : Store → Nat → α → Store
  byte : Store → Nat → Byte
  read_same : ∀ m a v, read (write m a v) a=v
  read_apart : ∀ m a b v, Apart a b → read (write m a v) b=read m b
  byte_frame : ∀ m a i v, (i < a  ∨  a+4 ≤ i) → byte (write m a v) i=byte m i

structure Write (α : Type) where
  address : Nat
  value : α

def run (api : API α Store Byte) (events : List (Write α)) (m : Store) : Store :=
  events.foldl (fun state e => api.write state e.address e.value) m

theorem run_read_frame (api : API α Store Byte) (events : List (Write α))
    (m : Store) (a : Nat) (h : ∀ e∈events, Apart e.address a) :
    api.read (run api events m) a=api.read m a := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    change api.read (run api es (api.write m e.address e.value)) a=_
    rw [ih _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
    exact api.read_apart m _ _ _ (h e (List.mem_cons_self ..))

theorem run_byte_frame (api : API α Store Byte) (events : List (Write α))
    (m : Store) (i : Nat) (h : ∀ e∈events, i < e.address  ∨  e.address+4 ≤ i) :
    api.byte (run api events m) i=api.byte m i := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    change api.byte (run api es (api.write m e.address e.value)) i=_
    rw [ih _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
    exact api.byte_frame m _ _ _ (h e (List.mem_cons_self ..))

/-- The final write to a selected address is read back, with later disjoint
writes proved harmless. This is a local store-trace lemma, not an output law. -/
theorem run_readback (api : API α Store Byte) (events : List (Write α))
    (m : Store) (a : Nat) (v : α)
    (present : ∃ e∈events, e.address=a  ∧  e.value=v)
    (only : ∀ e∈events, e.address=a  ∧  e.value=v  ∨  Apart e.address a) :
    api.read (run api events m) a=v := by
  induction events generalizing m with
  | nil => simp at present
  | cons e es ih =>
    change api.read (run api es (api.write m e.address e.value)) a=v
    by_cases tail : ∃ q∈es, q.address=a  ∧  q.value=v
    · exact ih _ tail (fun q hq => only q (List.mem_cons_of_mem _ hq))
    · have head : e.address=a  ∧  e.value=v := by
        obtain ⟨q,hq,ha,hv⟩ := present
        rcases List.mem_cons.mp hq with he|ht
        · simpa [he] using And.intro ha hv
        · exact False.elim (tail ⟨q,ht,ha,hv⟩)
      rw [run_read_frame api es _ a]
      · rw [head.1,head.2,api.read_same]
      · intro q hq
        rcases only q (List.mem_cons_of_mem _ hq) with hsame|apart
        · exact False.elim (tail ⟨q,hq,hsame⟩)
        · exact apart

/-- Arbitrary Real physical memory at byte starts. The frame is stated only
outside complete write windows, matching the byte-addressed native contract. -/
def realAPI : API ℝ (Nat → ℝ) ℝ where
  read := fun m i => m i
  write := fun m i v => Function.update m i v
  byte := fun m i => m i
  read_same := by intros; simp
  read_apart := by
    intro m a b v h
    exact Function.update_of_ne h.ne.symm v m
  byte_frame := by
    intro m a i v h
    apply Function.update_of_ne
    omega

#axiomsClean run_read_frame
#axiomsClean run_byte_frame
#axiomsClean run_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Memory
end VeriTile.Bench.Repairs.LayernormFwdTriton
