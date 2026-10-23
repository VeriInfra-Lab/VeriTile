import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Arithmetic
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Bounds
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Schedule

namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

abbrev Address := Nat × Nat
abbrev Memory (α : Type) := Address → α
structure Buffers where
  x : Nat
  w : Nat
  bias : Option Nat
  saved : Option Nat
  out : Nat
  deriving DecidableEq

def Buffers.reads (b : Buffers) (s : Shape) (p : Address) : Prop :=
  (0<s.k ∧ (p.1=b.x ∨ p.1=b.w)) ∨ b.bias=some p.1
def Buffers.writesStorage (b : Buffers) (p : Address) : Prop :=
  p.1=b.out ∨ b.saved=some p.1

def Separate (s : Shape) (b : Buffers) : Prop :=
  (∀ p,b.reads s p → ¬b.writesStorage p) ∧ b.saved≠some b.out

def outputOffset (s : Shape) (m n : Nat) := m*max s.n 1+n

def preactivation (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (memory : Memory α) (m n : Nat) : α :=
  accumulation a c mode s.k
    (fun k => a.decode mode (memory (b.x,m*max s.k 1+k)))
    (fun k => a.decode mode (memory (b.w,n*max s.k 1+k)))
    (b.bias.map (fun p => a.decode mode (memory (p,n))))

/-- The original K heuristic is true for exact multiples, and that arm is
unmasked. The remaining count is K-block*BK, with source pointers advanced. -/
def sourceOperand (a : Arithmetic α) (mode : Mode) (K bk block laneOffset : Nat)
    (memory : Memory α) (buffer row : Nat) : α :=
  if K%bk=0 ∨ laneOffset<K-block*bk then
    a.decode mode (memory (buffer,row*max K 1+block*bk+laneOffset))
  else a.literal .zero

def sourceBlock (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (memory : Memory α) (m n block : Nat) (z : α) : α :=
  let kind := selectedInstruction c mode
  foldN (fun group seed => a.instruction kind
    (fun lane => sourceOperand a mode s.k (configs c).bk block (group*kind.width+lane.val) memory b.x m)
    (fun lane => sourceOperand a mode s.k (configs c).bk block (group*kind.width+lane.val) memory b.w n)
    seed) ((configs c).bk/kind.width) z

def sourceAccumulation (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (memory : Memory α) (m n : Nat) : α :=
  let seed := match b.bias with
    | none => a.literal .zero
    | some p => a.add (a.decode mode (memory (p,n))) (a.literal .zero)
  foldN (sourceBlock a s b c mode memory m n) (s.blocks c) seed

 theorem foldN_bounded (f g : Nat → α → α) (n : Nat) (z : α)
    (h : ∀ j,j<n → ∀ x,f j x=g j x) : foldN f n z=foldN g n z := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [foldN,foldN,h n (by omega),ih (by intro j hj x; exact h j (by omega) x)]

 theorem sourceBlock_eq (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (memory : Memory α) (m n block : Nat) (z : α)
    (ht : block<s.blocks c) :
    sourceBlock a s b c mode memory m n block z=
      blockStep a c mode s.k block
        (fun k => a.decode mode (memory (b.x,m*max s.k 1+k)))
        (fun k => a.decode mode (memory (b.w,n*max s.k 1+k))) z := by
  have bk : 0<(configs c).bk := by have := config_bounds c; omega
  simp only [sourceBlock,blockStep]
  apply foldN_bounded
  intro group hg seed
  unfold instructionStep
  congr 1 <;> funext lane
  all_goals
    have hsize := instruction_divides c mode
    have hp := Nat.mul_le_mul_right (selectedInstruction c mode).width (Nat.succ_le_of_lt hg)
    have hl : group*(selectedInstruction c mode).width+lane.val<(configs c).bk := by
      have := lane.isLt
      nlinarith
    have hm := source_k_mask s.k (configs c).bk block
      (group*(selectedInstruction c mode).width+lane.val) bk ht hl
    simp only [sourceOperand,hm,Nat.add_assoc]

 theorem sourceAccumulation_eq (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (memory : Memory α) (m n : Nat) :
    sourceAccumulation a s b c mode memory m n=preactivation a s b c mode memory m n := by
  simp only [sourceAccumulation,preactivation,accumulation]
  have seed : (match b.bias with
      | none => a.literal .zero
      | some p => a.add (a.decode mode (memory (p,n))) (a.literal .zero))=
      (match b.bias.map (fun p => a.decode mode (memory (p,n))) with
      | none => a.literal .zero
      | some value => a.add value (a.literal .zero)) := by cases b.bias <;> rfl
  rw [seed]
  apply foldN_bounded
  intro block ht z
  exact sourceBlock_eq a s b c mode memory m n block z ht

 theorem preactivation_reads (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (entry current : Memory α)
    (same : ∀ p,b.reads s p → current p=entry p) (m n : Nat) :
    preactivation a s b c mode current m n=preactivation a s b c mode entry m n := by
  have hbias : b.bias.map (fun p => a.decode mode (current (p,n)))=
      b.bias.map (fun p => a.decode mode (entry (p,n))) := by
    cases hb : b.bias with
    | none => rfl
    | some p => simp only [Option.map_some]; rw [same _ (Or.inr hb)]
  by_cases hk : s.k=0
  · have bk : 0<(configs c).bk := by have := config_bounds c; omega
    simp only [preactivation,hbias,accumulation,hk,ceil_zero _ bk,foldN]
  · have positive : 0<s.k := by omega
    have hx : (fun k => a.decode mode (current (b.x,m*max s.k 1+k)))=
        (fun k => a.decode mode (entry (b.x,m*max s.k 1+k))) := by
      funext k; rw [same _ (Or.inl ⟨positive,Or.inl rfl⟩)]
    have hw : (fun k => a.decode mode (current (b.w,n*max s.k 1+k)))=
        (fun k => a.decode mode (entry (b.w,n*max s.k 1+k))) := by
      funext k; rw [same _ (Or.inl ⟨positive,Or.inr rfl⟩)]
    unfold preactivation
    rw [hx,hw,hbias]

#axiomsClean sourceBlock_eq
#axiomsClean sourceAccumulation_eq
#axiomsClean preactivation_reads
end
end VeriTile.Bench.Repairs.TritonLinearActivation
