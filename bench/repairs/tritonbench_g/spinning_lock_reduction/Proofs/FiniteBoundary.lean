import bench.repairs.tritonbench_g.spinning_lock_reduction.Formula
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.KernelStages

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.FiniteBoundary
open VeriTile.Bench.TritonBenchG.SpinningLock

/-- Shrink only the logical P/lock windows to the prefixes actually addressed
by this source. Original allocation/metadata restrictions remain available. -/
def activeBuffers (h : HostArgs) (buf : Wide.Buffers h) : Wide.Buffers h :=
  { buf with
    pElements := Formula.partialCount h
    lockElements := h.numSms
    partialCapacity := le_rfl
    lockCapacity := le_rfl
    partialStorageBytes := by have hc := buf.partialCapacity; have hb := buf.partialStorageBytes; dsimp [Formula.partialCount]; omega
    lockStorageBytes := by have hc := buf.lockCapacity; have hb := buf.lockStorageBytes; omega
    disjointPC := by
      have hc := buf.partialCapacity
      rcases buf.disjointPC with hl | hr
      · left; dsimp [Formula.partialCount]; omega
      · right; exact hr
    disjointPL := by
      have hp := buf.partialCapacity; have hk := buf.lockCapacity
      rcases buf.disjointPL with hl | hr
      · left; dsimp [Formula.partialCount]; omega
      · right; omega
    disjointCL := by
      have hk := buf.lockCapacity
      rcases buf.disjointCL with hl | hr
      · left; exact hl
      · right; omega }

def extend (default : A) (xs : Fin n → A) (address : Int) : A :=
  if ha : 0 ≤ address ∧ address < n then xs ⟨address.toNat,by omega⟩ else default

@[simp] theorem extend_at (default : A) (xs : Fin n → A) (i : Fin n) :
    extend default xs i.val = xs i := by
  simp [extend,i.isLt]

def memory (b : Backend V) (h : HostArgs) (input : Formula.Values V h outputCount) : Memory V :=
  ⟨extend b.zero input.partials,extend b.zero input.output,extend 0 input.locks⟩

def snapshot (h : HostArgs) (outputCount : Nat) (mem : Memory V) : Formula.Values V h outputCount :=
  ⟨fun i => mem.partials i.val,fun i => mem.output i.val,fun i => mem.locks i.val⟩

theorem memory_published (b : Backend V) (h : HostArgs)
    (input : Formula.Values V h outputCount) (published : Formula.Published b.zero h input) :
    Wide.PublishedReady b h (memory b h input) := by
  intro q ready lane hl
  change lane < h.bm*h.bn at hl
  by_cases hq : q < h.numSms
  · have lock : input.locks ⟨q,hq⟩ = 1 := by simpa [memory,extend,hq] using ready
    have hh := published ⟨q,hq⟩ lock lane hl
    have bound : q*h.bm*h.bn+lane < Formula.partialCount h := by
      have hp := Nat.mul_le_mul_right (h.bm*h.bn) (Nat.succ_le_of_lt hq)
      simp only [Nat.succ_mul,Nat.mul_assoc,Formula.partialCount] at hp ⊢
      omega
    rw [show Wide.pAddress h q lane = ((q*h.bm*h.bn+lane : Nat) : Int) by simp [Wide.pAddress]]
    exact (extend_at b.zero input.partials ⟨q*h.bm*h.bn+lane,bound⟩).trans hh
  · have impossible : (0 : I32) = 1 := by simpa [memory,extend,hq] using ready
    exact False.elim ((by decide : (0 : I32) ≠ 1) impossible)

theorem memory_ready (b : Backend V) (h : HostArgs)
    (input : Formula.Values V h outputCount) (ready : Formula.Ready h input) :
    KernelProof.ConsumerReady h (memory b h input) := by
  intro p hp role
  simpa [memory,extend,hp] using ready ⟨p,hp⟩ role

theorem snapshot_published (b : Backend V) (h : HostArgs) (outputCount : Nat)
    (mem : Memory V) (published : Wide.PublishedReady b h mem) :
    Formula.Published b.zero h (snapshot h outputCount mem) := by
  intro p ready lane hl
  simpa [snapshot,Wide.pAddress] using published p.val ready lane hl

end VeriTile.Bench.Repairs.SpinningLock.FiniteBoundary
