import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.IO
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.SharedReadback

namespace VeriTile.Bench.Repairs.KvCacheFilling.Interface
open VeriTile.Triton FP.Structural QuantProfile CopyExecution QuantExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000

def sourceRowTarget {α : Type} [Inhabited α] (a : Ops α) (r : Request) (s : State α) (src : Nat)
    (key : Bool) :=
  SharedTarget.row a r.policy (capacity (Shared.parameters r) key)
    (lanes (Shared.parameters r) key) (inputRow (Shared.parameters r) s src key)

theorem cache_cell {α : Type} [Inhabited α] (a : Ops α) (r : Request) (d : Domain r)
    (s f : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (version : Bool) (success : exec (algebra a) (Shared.kernel version r) s = some f)
    (key : Bool) (i : Nat) (hi : i < capacity (Shared.parameters r) key) :
    f.mem (output key) (dst*capacity (Shared.parameters r) key+i) =
      FP.Structural.Cell.mk (cacheDType r.policy)
        (cacheValue r.policy ((sourceRowTarget a r s src key).value i)) := by
  have pos := SharedExecution.positive r d
  have cov := SharedExecution.covered r
  cases hp : r.policy with
  | copy =>
    have shape := SharedReadback.single_shape r d (by simp [hp])
    have shp : ∀ key, capacity (Shared.parameters r) key = width (Shared.parameters r) key := by
      intro key; cases key <;> simp [capacity,width,Shared.parameters,shape.1,shape.2]
    have covered : ∀ key, width (Shared.parameters r) key ≤ lanes (Shared.parameters r) key := by
      intro key; rw [← shp key]; exact cov key
    have ex := CopyExecution.execute (algebra a) (Shared.parameters r) s src dst hs hd version
    have same : f = CopyExecution.result (algebra a) (Shared.parameters r) s src dst := by
      apply Option.some.inj
      exact success.symm.trans (by simpa [Shared.kernel,hp] using ex)
    subst f
    simpa [sourceRowTarget,SharedTarget.row,hp,cacheDType,cacheValue,inputRow,algebra,shp] using
      CopyExecution.result_read (algebra a) (Shared.parameters r) s src dst shp covered key i
        (by rwa [← shp key])
  | int4 =>
    let e := QuantExecution.executed4 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 version
    have same : f = e.val := by
      apply Option.some.inj
      exact success.symm.trans (by simpa [Shared.kernel,hp] using e.property)
    rw [same]
    have obs := QuantValues.observe4 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 cov version
    simpa [sourceRowTarget,SharedTarget.row,hp,cacheDType,cacheValue,QuantValues.rows4] using obs.cache key i hi
  | int8 =>
    have shape := SharedReadback.single_shape r d (by simp [hp])
    let e := QuantExecution.executed8 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 version
    have same : f = e.val := by
      apply Option.some.inj
      exact success.symm.trans (by simpa [Shared.kernel,hp] using e.property)
    rw [same]
    have obs := QuantValues.observe8 a (Shared.parameters r) s src dst hs hd pos.1 pos.2
      shape.1 shape.2 cov version
    simpa [sourceRowTarget,SharedTarget.row,hp,cacheDType,cacheValue,QuantValues.rows8] using obs.cache key i hi

theorem metadata_cell {α : Type} [Inhabited α] (a : Ops α) (r : Request) (d : Domain r)
    (s f : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (version : Bool) (success : exec (algebra a) (Shared.kernel version r) s = some f)
    (hp : r.policy ≠ .copy) (key : Bool) (i : Nat) (hi : i < 2) :
    f.mem (metaRegion key) (dst*2+i) = FP.Structural.Cell.mk .real
      (if i = 0 then (sourceRowTarget a r s src key).scale else (sourceRowTarget a r s src key).zero) := by
  have pos := SharedExecution.positive r d
  have cov := SharedExecution.covered r
  cases pol : r.policy with
  | copy => exact False.elim (hp pol)
  | int4 =>
    let e := QuantExecution.executed4 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 version
    have same : f = e.val := by
      apply Option.some.inj
      exact success.symm.trans (by simpa [Shared.kernel,pol] using e.property)
    rw [same]
    have obs := QuantValues.observe4 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 cov version
    simpa [sourceRowTarget,SharedTarget.row,pol,QuantValues.rows4] using obs.metadata key i hi
  | int8 =>
    have shape := SharedReadback.single_shape r d (by simp [pol])
    let e := QuantExecution.executed8 a (Shared.parameters r) s src dst hs hd pos.1 pos.2 version
    have same : f = e.val := by
      apply Option.some.inj
      exact success.symm.trans (by simpa [Shared.kernel,pol] using e.property)
    rw [same]
    have obs := QuantValues.observe8 a (Shared.parameters r) s src dst hs hd pos.1 pos.2
      shape.1 shape.2 cov version
    simpa [sourceRowTarget,SharedTarget.row,pol,QuantValues.rows8] using obs.metadata key i hi

end
end VeriTile.Bench.Repairs.KvCacheFilling.Interface
