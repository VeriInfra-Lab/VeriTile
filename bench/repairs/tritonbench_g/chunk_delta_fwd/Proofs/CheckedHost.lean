import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.ReturnedMemory
import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.NativePointers

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked.HostRun
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000

inductive Stage where | admission | allocation | pointers | setup deriving DecidableEq
structure Failure where
  stage : Stage
  currentDevice : Nat
structure Returned where
  tensors : Tensor×Tensor
  memory : Bytes.Memory
  currentDevice : Nat
  launchDevice : Option Nat

/-- Only ordinary allocation metadata and a Unit-valued setup port are
external. No source execution, dot value, or output enters through a port. -/
structure Runtime (r : Request) where
  allocate : Except String (Allocations r)
  warps : Nat
  selected : warps∈([1,2,4,8,16,32] : List Nat)
  setup : Allocations r → Nat → Nat → Except String Unit

def result (I : Primitives) (r : Request) (a : Allocations r) (before : Bytes.Memory)
    (caller : Nat) : Returned :=
  ⟨(a.history,a.corrected),returnedBytes I r a before,caller,
    if r.B*r.H*r.V=0 then none else some r.key.device⟩

def run (I : Primitives) (r : Request) (before : Bytes.Memory) (rt : Runtime r) (caller : Nat) :
    Except Failure Returned :=
  if r.check then
    match rt.allocate with
    | .error _=>.error ⟨.allocation,caller⟩
    | .ok a=>if a.check then
        if r.B*r.H*r.V=0 then .ok (result I r a before caller)
        else match rt.setup a r.key.device rt.warps with
          | .error _=>.error ⟨.setup,caller⟩
          | .ok _=>.ok (result I r a before caller)
      else .error ⟨.pointers,caller⟩
  else .error ⟨.admission,caller⟩

structure Success (I : Primitives) (r : Request) (before : Bytes.Memory) (rt : Runtime r)
    (caller : Nat) (returned : Returned) : Prop where
  checked : r.check=Bool.true
  allocation : ∃ a : Allocations r,rt.allocate=.ok a ∧ a.check=Bool.true ∧
    (r.B*r.H*r.V≠0 → rt.setup a r.key.device rt.warps=.ok ()) ∧ returned=result I r a before caller

def Spec (I : Primitives) (r : Request) (before : Bytes.Memory) (rt : Runtime r) (caller : Nat) :
    Except Failure Returned → Prop
  | .error f=>f.currentDevice=caller
  | .ok returned=>Success I r before rt caller returned

theorem complete (I : Primitives) (r : Request) (before : Bytes.Memory) (rt : Runtime r) (caller : Nat) :
    Spec I r before rt caller (run I r before rt caller) := by
  unfold run
  split
  · rename_i checked
    cases allocation : rt.allocate with
    | error reason=>rfl
    | ok a=>
      simp only
      split
      · rename_i pointers
        split
        · rename_i empty
          exact ⟨checked,a,allocation,pointers,fun h=>False.elim (h empty),rfl⟩
        · rename_i nonempty
          cases setup : rt.setup a r.key.device rt.warps with
          | error reason=>rfl
          | ok token=>
            cases token
            exact ⟨checked,a,allocation,pointers,fun _=>setup,rfl⟩
      · rfl
  · rfl

theorem returned (I : Primitives) (r : Request) (before : Bytes.Memory) (rt : Runtime r) (caller : Nat)
    (out : Returned) (success : run I r before rt caller=.ok out) : Success I r before rt caller out := by
  have h:=complete I r before rt caller
  rw [success] at h
  exact h

theorem pointer (r : Request) (checked : r.check=Bool.true) (a : Allocations r)
    (pointers : a.check=Bool.true) (region : RegionName) (t : Tensor) (bound : descriptor r a region=some t) :
    Pointer t (formedExtent r.dims) := by
  have hp : Pointer a.history (formedExtent r.dims) ∧ Pointer a.corrected (formedExtent r.dims):=by
    simpa [Allocations.check] using pointers
  have original:=r.pointers checked
  unfold descriptor at bound
  split at bound
  · cases bound;exact original r.key (by simp [Request.all,Request.inputs])
  · split at bound
    · cases bound;exact original r.weight (by simp [Request.all,Request.inputs])
    · split at bound
      · cases bound;exact original r.value (by simp [Request.all,Request.inputs])
      · split at bound
        · exact original t (by simp [Request.all,Request.inputs,bound])
        · split at bound
          · cases bound;exact hp.1
          · split at bound
            · cases bound;exact hp.2
            · split at bound
              · exact original t (by simp [Request.all,bound])
              · contradiction

def native_addresses (r : Request) (checked : r.check=Bool.true) (a : Allocations r)
    (pointers : a.check=Bool.true) (x : NativePointers.Coordinates r.dims) (site : NativePointers.Site)
    (t : Tensor) (bound : descriptor r a (NativePointers.region site)=some t) :=
  NativePointers.native r checked x site t (pointer r checked a pointers _ t bound)

#axiomsClean complete
#axiomsClean pointer
#axiomsClean native_addresses
#specNonCircular Runtime avoiding [run,FP.Structural.exec,runGrid,source,gridMemory,states]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked.HostRun
