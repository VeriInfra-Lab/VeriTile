import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.SharedExecution

namespace VeriTile.Bench.Repairs.KvCacheFilling.SharedReadback
open VeriTile.Triton FP.Structural QuantProfile CopyExecution QuantExecution QuantValues
noncomputable section

theorem single_shape (r : Request) (d : Domain r) (h : r.policy ≠ .int4) :
    r.dk = r.ck ∧ r.dv = r.cv := SharedExecution.single_shape r d h

def Outside (r : Request) (dst : Nat) (reg : RegionName) (i : Nat) : Prop :=
  (∀ key, reg ≠ output key ∨ i < dst*capacity (Shared.parameters r) key ∨
    dst*capacity (Shared.parameters r) key+capacity (Shared.parameters r) key ≤ i) ∧
  (r.policy ≠ .copy → ∀ key, reg ≠ metaRegion key ∨ i < dst*2 ∨ dst*2+2 ≤ i)

theorem frame {α : Type} [Inhabited α] (a : Ops α) (r : Request) (d : Domain r)
    (s f : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int)) (version : Bool)
    (success : exec (algebra a) (Shared.kernel version r) s = some f)
    (reg : RegionName) (i : Nat) (outside : Outside r dst reg i) : f.mem reg i = s.mem reg i := by
  have pos := SharedExecution.positive r d
  cases hp : r.policy with
  | copy =>
    have sh := single_shape r d (by simp [hp])
    have shp : ∀ key, capacity (Shared.parameters r) key = width (Shared.parameters r) key := by
      intro key; cases key <;> simp [capacity,width,Shared.parameters,sh.1,sh.2]
    have ex := CopyExecution.execute (algebra a) (Shared.parameters r) s src dst hs hd version
    have same : f = CopyExecution.result (algebra a) (Shared.parameters r) s src dst :=
      Option.some.inj (success.symm.trans (by simpa [Shared.kernel,hp] using ex))
    rw [same]
    exact CopyExecution.result_frame (algebra a) (Shared.parameters r) s src dst shp reg i
      (by simpa only [shp] using outside.1)
  | int4 =>
    have ex := execute4 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 version
    have same : f = result4 a (Shared.parameters r) s src dst :=
      Option.some.inj (success.symm.trans (by simpa [Shared.kernel,hp] using ex))
    rw [same]
    exact finished_frame (Shared.parameters r) dst (rows4 a (Shared.parameters r) s src)
      (prepared4 a (Shared.parameters r) s src dst) reg i outside.1 (outside.2 (by simp [hp]))
  | int8 =>
    have sh := single_shape r d (by simp [hp])
    have ex := execute8 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 sh.1 sh.2 version
    have same : f = result8 a (Shared.parameters r) s src dst :=
      Option.some.inj (success.symm.trans (by simpa [Shared.kernel,hp] using ex))
    rw [same]
    exact finished_frame (Shared.parameters r) dst (rows8 a (Shared.parameters r) s src)
      (prepared8 a (Shared.parameters r) s src dst) reg i outside.1 (outside.2 (by simp [hp]))

#axiomsClean frame
end
end VeriTile.Bench.Repairs.KvCacheFilling.SharedReadback
