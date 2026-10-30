import bench.tritonbench_g.kv_cache_filling.Proofs.QuantValues
import bench.tritonbench_g.kv_cache_filling.Proofs.SharedTarget

namespace VeriTile.Bench.Repairs.KvCacheFilling.SharedExecution
open VeriTile.Triton FP.Structural QuantProfile CopyExecution QuantExecution
noncomputable section

theorem covered (r : Request) : ∀ key,
    capacity (Shared.parameters r) key ≤ lanes (Shared.parameters r) key := by
  intro key
  cases key <;> exact SharedTarget.block_covers _

theorem positive (r : Request) (_d : Domain r) :
    0 < (Shared.parameters r).bk ∧ 0 < (Shared.parameters r).bv := by
  constructor <;> exact Nat.pos_of_isPowerOfTwo (Nat.isPowerOfTwo_nextPowerOfTwo _)

theorem single_shape (r : Request) (d : Domain r) (h : r.policy ≠ .int4) :
    r.dk = r.ck ∧ r.dv = r.cv := by
  constructor
  · simpa [Request.factor,h] using d.key_shape
  · simpa [Request.factor,h] using d.value_shape

/-- The witness is produced from source execution, with input-only schedule
premises and the public shape contract. -/
def executed {α : Type} [Inhabited α] (a : Ops α) (r : Request) (d : Domain r)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int)) (version : Bool) :
    { f : State α // exec (algebra a) (Shared.kernel version r) s = some f } := by
  have pos := positive r d
  cases hp : r.policy with
  | copy =>
    refine ⟨CopyExecution.result (algebra a) (Shared.parameters r) s src dst,?_⟩
    simpa [Shared.kernel,hp] using CopyExecution.execute (algebra a) (Shared.parameters r) s src dst hs hd version
  | int4 =>
    refine ⟨result4 a (Shared.parameters r) s src dst,?_⟩
    simpa [Shared.kernel,hp] using execute4 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 version
  | int8 =>
    have sh := single_shape r d (by simp [hp])
    refine ⟨result8 a (Shared.parameters r) s src dst,?_⟩
    simpa [Shared.kernel,hp] using execute8 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 sh.1 sh.2 version

#axiomsClean executed
end
end VeriTile.Bench.Repairs.KvCacheFilling.SharedExecution
