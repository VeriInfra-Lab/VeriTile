import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Arithmetic
import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Schedule
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Addresses are typed scalar observations. Storage identity is shared across
views, while element offsets retain the source dtype's units. This is not an
independent physical-byte execution interface. -/
inductive DType where | fp32 | uint8 | int32 deriving DecidableEq
abbrev Region := Nat × DType
abbrev Address := Region × Nat
abbrev Memory (α : Type) := Address → α

structure Buffers where
  a : Nat
  b : Nat
  out : Nat
  a0 : Nat
  b0 : Nat
  am : Nat
  ak : Nat
  bk : Nat
  bn : Nat
  deriving DecidableEq

def Buffers.aRegion (b : Buffers) : Region := (b.a,.fp32)
def Buffers.bRegion (b : Buffers) : Region := (b.b,.uint8)
def Buffers.outRegion (b : Buffers) : Region := (b.out,.int32)
def aAddress (b : Buffers) (m k : Nat) : Address := (b.aRegion,b.a0+m*b.am+k*b.ak)
def bAddress (b : Buffers) (k n : Nat) : Address := (b.bRegion,b.b0+k*b.bk+n*b.bn)
def outputOffset (s : Shape) (m n : Nat) := m*max s.n 1+n
def outAddress (s : Shape) (b : Buffers) (m n : Nat) : Address := (b.outRegion,outputOffset s m n)

def Buffers.reads (b : Buffers) (loc : Address) : Prop := loc.1=b.aRegion ∨ loc.1=b.bRegion
def Separate (b : Buffers) : Prop := b.out≠b.a ∧ b.out≠b.b

theorem separate_read_write (b : Buffers) (h : Separate b) (p : Address)
    (read : b.reads p) : p.1≠b.outRegion := by
  rcases read with ha|hb
  · rw [ha]
    intro he
    have := congrArg Prod.fst he
    exact h.1 this.symm
  · rw [hb]
    intro he
    have := congrArg Prod.fst he
    exact h.2 this.symm

/-- These masks and continuously advancing A / reset B offsets are the source
expressions. Masked zero is unused in every legal nonzero loop iteration. -/
def sourceA (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (f j lane m : Nat) (memory : Memory α) : Byte :=
  if lane<s.k-(f*s.steps c+j)*(configs c).bk then
    a.cast (memory (aAddress b m ((f*s.steps c+j)*(configs c).bk+lane))) else 0

def sourceB (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (j lane n : Nat) (memory : Memory α) : Byte :=
  if lane<s.k then a.packed (memory (bAddress b (j*(configs c).bk+lane) n)) else 0

def sourceBlock (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (f j m n : Nat) (memory : Memory α) (seed : β) : β :=
  foldN (fun group z => a.instruction (selectedInstruction c)
    (fun lane : Fin 32 => sourceA a s b c f j (group*32+lane.val) m memory)
    (fun lane : Fin 32 => weightByte (sourceB a s b c j (group*32+lane.val) n memory) f) z)
    ((configs c).bk/32) seed

def sourceAccumulation (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (memory : Memory α) (m n : Nat) : β :=
  foldN (fun f z => foldN (fun j seed => sourceBlock a s b c f j m n memory seed) (s.steps c) z) 4 a.zero

def callerAccumulation (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (memory : Memory α) (m n : Nat) : β :=
  accumulation a s c (fun k => memory (aAddress b m k)) (fun k => memory (bAddress b k n))

theorem foldN_congr_bound (f g : Nat → β → β) (n : Nat)
    (h : ∀ i,i<n → ∀ z,f i z=g i z) (z : β) : foldN f n z=foldN g n z := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [foldN,foldN,ih (by intro i hi; exact h i (by omega)),h n (by omega)]

theorem group_lane_bound (c : Fin 15) (group : Nat) (hg : group<(configs c).bk/32) (lane : Fin 32) :
    group*32+lane.val<(configs c).bk := by
  have h := Nat.mul_le_mul_right 32 (Nat.succ_le_of_lt hg)
  have d := Nat.div_mul_le_self (configs c).bk 32
  have l := lane.isLt
  omega

theorem sourceBlock_eq (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (legal : s.legal c) (f j m n : Nat) (hf : f<4) (hj : j<s.steps c)
    (memory : Memory α) (seed : β) :
    sourceBlock a s b c f j m n memory seed=
      blockStep a c (s.steps c) f j (fun k => memory (aAddress b m k))
        (fun k => memory (bAddress b k n)) seed := by
  unfold sourceBlock blockStep
  apply foldN_congr_bound
  intro group hg z
  unfold instructionStep
  congr 1
  · funext lane
    have masks := legal_all_masks s c legal f j (group*32+lane.val) hf hj (group_lane_bound c group hg lane)
    simp only [sourceA,if_pos masks.1,Nat.add_assoc]
  · funext lane
    have masks := legal_all_masks s c legal f j (group*32+lane.val) hf hj (group_lane_bound c group hg lane)
    simp only [sourceB,if_pos masks.2.1,Nat.add_assoc]

theorem sourceAccumulation_eq (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (legal : s.legal c) (memory : Memory α) (m n : Nat) :
    sourceAccumulation a s b c memory m n=callerAccumulation a s b c memory m n := by
  unfold sourceAccumulation callerAccumulation accumulation
  apply foldN_congr_bound
  intro f hf z
  unfold fieldStep
  apply foldN_congr_bound
  intro j hj seed
  exact sourceBlock_eq a s b c legal f j m n hf hj memory seed

theorem accumulation_reads (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (entry current : Memory α) (same : ∀ p,b.reads p → current p=entry p) (m n : Nat) :
    callerAccumulation a s b c current m n=callerAccumulation a s b c entry m n := by
  have ha : (fun k => current (aAddress b m k))=(fun k => entry (aAddress b m k)) := by
    funext k
    exact same _ (Or.inl rfl)
  have hb : (fun k => current (bAddress b k n))=(fun k => entry (bAddress b k n)) := by
    funext k
    exact same _ (Or.inr rfl)
  unfold callerAccumulation
  rw [ha,hb]

#axiomsClean sourceBlock_eq
#axiomsClean sourceAccumulation_eq
#axiomsClean accumulation_reads
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
