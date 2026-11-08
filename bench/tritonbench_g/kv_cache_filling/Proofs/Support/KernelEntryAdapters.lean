import bench.tritonbench_g.kv_cache_filling.Proofs.Support.Model
import bench.tritonbench_g.kv_cache_filling.Kernels

namespace VeriTile.Bench.Repairs.KvCacheFilling.CopySource
open VeriTile.Triton
noncomputable section

structure Parameters where
  (dk dv ck cv bk bv : Nat)

def originalKernel (p : Parameters) : ComputeKernel :=
  original0Kernel "key" "value" "keyCache" "valueCache" "keyMeta" "valueMeta" "src" "dst" p.dk p.dv p.ck p.cv p.bk p.bv

def commutedKernel (p : Parameters) : ComputeKernel :=
  candidate0Kernel "key" "value" "keyCache" "valueCache" "keyMeta" "valueMeta" "src" "dst" p.dk p.dv p.ck p.cv p.bk p.bv

def kernel (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commutedKernel p else originalKernel p

end
end VeriTile.Bench.Repairs.KvCacheFilling.CopySource

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantSource
open VeriTile.Triton
noncomputable section
abbrev Parameters := CopySource.Parameters

def original8Kernel (p : Parameters) : ComputeKernel :=
  VeriTile.Bench.Repairs.KvCacheFilling.original8Kernel "key" "value" "keyCache" "valueCache" "keyMeta" "valueMeta" "src" "dst" p.dk p.dv p.ck p.cv p.bk p.bv

def commuted8Kernel (p : Parameters) : ComputeKernel :=
  candidate8Kernel "key" "value" "keyCache" "valueCache" "keyMeta" "valueMeta" "src" "dst" p.dk p.dv p.ck p.cv p.bk p.bv

def kernel8 (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commuted8Kernel p else original8Kernel p

def original4Kernel (p : Parameters) : ComputeKernel :=
  VeriTile.Bench.Repairs.KvCacheFilling.original4Kernel "key" "value" "keyCache" "valueCache" "keyMeta" "valueMeta" "src" "dst" p.dk p.dv p.ck p.cv p.bk p.bv

def commuted4Kernel (p : Parameters) : ComputeKernel :=
  candidate4Kernel "key" "value" "keyCache" "valueCache" "keyMeta" "valueMeta" "src" "dst" p.dk p.dv p.ck p.cv p.bk p.bv

def kernel4 (version : Bool) (p : Parameters) : ComputeKernel :=
  if version then commuted4Kernel p else original4Kernel p

end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantSource

namespace VeriTile.Bench.Repairs.KvCacheFilling.Shared
open VeriTile.Triton

def parameters (r : Request) : CopySource.Parameters :=
  ⟨r.dk, r.dv, r.ck, r.cv, r.ck.nextPowerOfTwo, r.cv.nextPowerOfTwo⟩

def kernel (version : Bool) (r : Request) : ComputeKernel :=
  match r.policy with
  | .copy => CopySource.kernel version (parameters r)
  | .int4 => QuantSource.kernel4 version (parameters r)
  | .int8 => QuantSource.kernel8 version (parameters r)

end VeriTile.Bench.Repairs.KvCacheFilling.Shared
