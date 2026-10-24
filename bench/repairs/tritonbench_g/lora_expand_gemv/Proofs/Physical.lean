import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Native
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Host
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Bounds
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Physical
open VeriTile.Triton FP.Structural FPValue Native
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option maxHeartbeats 6000000
abbrev Byte := BitVec 8
abbrev Bytes := RegionName → Nat → Byte

def encode (x : Bits) (byte : Fin 4) : Byte := x.extractLsb' (8*byte.val) 8
def decode (bytes : Fin 4 → Byte) : Bits := bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0

theorem decode_encode (x : Bits) : decode (encode x)=x := by
  change (x.extractLsb' 24 8 ++ x.extractLsb' 16 8 ++ x.extractLsb' 8 8 ++ x.extractLsb' 0 8)=x
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (bytes : Fin 4 → Byte) : encode (decode bytes)=bytes := by
  funext byte
  fin_cases byte
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Bits).extractLsb' 0 8=bytes 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Bits).extractLsb' 8 8=bytes 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Bits).extractLsb' 16 8=bytes 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Bits).extractLsb' 24 8=bytes 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_eq_self

/-- Bytes are relative to each actual tensor data pointer, including its storage
 offset. Binding Y to caller storage is injective and disjoint from X/W/I.
These ordinary ABI ownership facts are separate from the exact stores below. -/
def load32 (m : Bytes) (r : RegionName) (a : Nat) : Bits := decode (fun i => m r (4*a+i.val))
def load16 (m : Bytes) (r : RegionName) (a : Nat) : Bits :=
  ((m r (2*a+1) ++ m r (2*a)) : BitVec 16).zeroExtend 32

def data (m : Bytes) (r : RegionName) (a : Nat) : Bits :=
  if r="W" then load16 m r a else load32 m r a

def indices (m : Bytes) (a : Nat) : Int := (load32 m "Iraw" a).toInt

def store (m : Bytes) (a : Nat) (value : Bits) : Bytes :=
  fun r b => if r="Y" ∧ b/4=a then encode value ⟨b%4,Nat.mod_lt _ (by decide)⟩ else m r b

def project (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits) (s : Memory) (b : Nat) : Byte :=
  encode (eval A S bits ((s "Y" (b/4)).read .real)) ⟨b%4,Nat.mod_lt _ (by decide)⟩

theorem entry_related (A : ScalarOps) (S : Schedule) (g : Geometry) (m : Bytes) (raw : Nat → Int) :
    ∀ b,project A S (data m) (entry g raw) b=m "Y" b := by
  intro b
  simp only [project,entry,show ("Y" : RegionName)≠"I" by decide,
    show ("Y" : RegionName)≠"W" by decide,ite_false,Cell.read_same,eval,data,ite_false,load32]
  rw [encode_decode]
  dsimp only
  congr 1
  omega

def active (g : Geometry) (s : Memory) (q : Nat) : Prop := (s "I" (q/g.N)).read .int≠-1

def written (plans : ∀ n, ReductionPlan n) (g : Geometry) (s : Memory) (q : Nat) : Cell Word :=
  Branches.outputCell .ieeeValue plans .singleHalf .single g.parameters
    (program g (q/g.N) (q%g.N) s) s 0

def step (plans : ∀ n, ReductionPlan n) (g : Geometry) (s : Memory) (q : Nat) : Memory :=
  if active g s q then put s (offset g q) (written plans g s q) else s

def both (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) : List Nat → Memory → Bytes → Memory × Bytes
  | [],s,m => (s,m)
  | q::qs,s,m => both A S bits plans g qs (step plans g s q)
      (if active g s q then store m (offset g q) (eval A S bits ((written plans g s q).read .real)) else m)

theorem step_related (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (s : Memory) (m : Bytes)
    (q : Nat) (h : ∀ b,project A S bits s b=m "Y" b) :
    ∀ b,project A S bits (step plans g s q) b=
      (if active g s q then store m (offset g q) (eval A S bits ((written plans g s q).read .real)) else m) "Y" b := by
  intro b
  by_cases ha : active g s q
  · by_cases ho : b/4=offset g q
    · simp [step,ha,put,project,store,ho]
    · simpa [step,ha,put,project,store,ho] using h b
  · simpa [step,ha] using h b

theorem both_related (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (qs : List Nat) (s : Memory) (m : Bytes)
    (h : ∀ b,project A S bits s b=m "Y" b) :
    ∀ b,project A S bits (both A S bits plans g qs s m).1 b=(both A S bits plans g qs s m).2 "Y" b := by
  induction qs generalizing s m with
  | nil => exact h
  | cons q qs ih => exact ih _ _ (step_related A S bits plans g s m q h)

theorem both_source (rest : Algebra Word) (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (qs : List Nat)
    (bounded : ∀ q∈qs,q<g.count) (s : Memory) (m : Bytes) :
    execute rest plans g qs s=some (both A S bits plans g qs s m).1 := by
  induction qs generalizing s m with
  | nil => rfl
  | cons q qs ih =>
    have hc := coordinates g q (bounded q (by simp))
    obtain ⟨next,he,hm,_⟩ := Branches.correctness rest .ieeeValue plans g.parameters
      (program g (q/g.N) (q%g.N) s) (by change 0<256; decide)
    have he' : FP.Structural.exec (Branches.engine rest .ieeeValue plans)
        (Branches.source .singleHalf .single g.parameters)
        (program g (q/g.N) (q%g.N) s)=some next := he
    have hs : next.mem=step plans g s q := by
      rw [hm,write_single plans g _ _ s hc.1]
      simp only [step,active,written,offset]
      split <;> simp_all
    rw [execute,he']
    simp only [Option.bind_some,hs,both]
    exact ih (fun p hp => bounded p (by simp [hp])) _ _

theorem both_frame (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (qs : List Nat) (s : Memory) (m : Bytes)
    (r : RegionName) (b : Nat) (h : r≠("Y" : RegionName) ∨ ∀ q∈qs,b/4≠offset g q) :
    (both A S bits plans g qs s m).2 r b=m r b := by
  induction qs generalizing s m with
  | nil => rfl
  | cons q qs ih =>
    rw [both,ih _ _ (h.imp_right (fun hh p hp => hh p (by simp [hp])))]
    split
    · have miss : ¬(r="Y" ∧ b/4=offset g q) := by
        intro eq; rcases h with h|h
        · exact h eq.1
        · exact h q (by simp) eq.2
      simp [store,miss]
    · rfl

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean entry_related
#axiomsClean step_related
#axiomsClean both_related
#axiomsClean both_source
#axiomsClean both_frame
#specNonCircular store avoiding [execute,Branches.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Physical
