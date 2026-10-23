import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Physical
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Caller
open VeriTile.Triton FP.Structural FPValue Native Physical
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option maxHeartbeats 4000000
abbrev Bytes := Nat → Nat → Byte

def bind (g : Geometry) (a : Host.ABI g) (m : Bytes) : Physical.Bytes := fun r b =>
  if r="X" then m a.x.allocation (a.x.byteBase+b)
  else if r="W" then m a.w.allocation (a.w.byteBase+b)
  else if r="Y" then m a.y.allocation (a.y.byteBase+b)
  else if r="Iraw" then m a.indices.allocation (a.indices.byteBase+b) else 0

/-- Actual caller-storage byte mutation, including the supplied Y's nonzero
storage/data-pointer offset. No whole-output map is assumed. -/
def store (g : Geometry) (a : Host.ABI g) (m : Bytes) (off : Nat) (v : Bits) : Bytes :=
  fun r b => if r=a.y.allocation ∧ a.y.byteBase≤b ∧ (b-a.y.byteBase)/4=off then
    encode v ⟨(b-a.y.byteBase)%4,Nat.mod_lt _ (by decide)⟩ else m r b

def run (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (a : Host.ABI g) : List Nat → Memory → Bytes → Bytes
  | [],_,m => m
  | q::qs,s,m => run A S bits plans g a qs (Physical.step plans g s q)
      (if Physical.active g s q then store g a m (offset g q)
        (eval A S bits ((Physical.written plans g s q).read .real)) else m)

theorem store_related (g : Geometry) (a : Host.ABI g) (m : Bytes) (v : Physical.Bytes)
    (h : ∀ b,m a.y.allocation (a.y.byteBase+b)=v "Y" b) (off : Nat) (w : Bits) :
    ∀ b,store g a m off w a.y.allocation (a.y.byteBase+b)=Physical.store v off w "Y" b := by
  intro b
  by_cases ho : b/4=off
  · simp [store,Physical.store,ho]
  · simpa [store,Physical.store,ho] using h b

theorem related (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (a : Host.ABI g)
    (qs : List Nat) (s : Memory) (m : Bytes) (v : Physical.Bytes)
    (h : ∀ b,m a.y.allocation (a.y.byteBase+b)=v "Y" b) :
    ∀ b,run A S bits plans g a qs s m a.y.allocation (a.y.byteBase+b)=
      (Physical.both A S bits plans g qs s v).2 "Y" b := by
  induction qs generalizing s m v with
  | nil => exact h
  | cons q qs ih =>
    apply ih
    by_cases ha : Physical.active g s q
    · simpa only [ha,ite_true] using store_related g a m v h (offset g q)
        (eval A S bits ((Physical.written plans g s q).read .real))
    · simpa only [ha,ite_false] using h

theorem frame (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (a : Host.ABI g)
    (qs : List Nat) (s : Memory) (m : Bytes) (r b : Nat)
    (h : r≠a.y.allocation ∨ b<a.y.byteBase ∨ ∀ q∈qs,(b-a.y.byteBase)/4≠offset g q) :
    run A S bits plans g a qs s m r b=m r b := by
  induction qs generalizing s m with
  | nil => rfl
  | cons q qs ih =>
    rw [run,ih _ _ (h.imp_right (fun hh => hh.imp_right (fun hh p hp => hh p (by simp [hp]))))]
    split
    · have miss : ¬(r=a.y.allocation ∧ a.y.byteBase≤b ∧ (b-a.y.byteBase)/4=offset g q) := by
        intro eq
        rcases h with h|h|h
        · exact h eq.1
        · omega
        · exact h q (by simp) eq.2.2
      simp [store,miss]
    · rfl

#axiomsClean store_related
#axiomsClean related
#axiomsClean frame
#specNonCircular store avoiding [Host.invoke,execute,Branches.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Caller
