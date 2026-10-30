import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.ReturnedMemory
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CheckedPrefixNative

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CheckedHost
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory HostStorage
noncomputable section
set_option maxHeartbeats 8000000

inductive Stage where | admission | prepare | pointers | setup | finish | empty deriving DecidableEq
structure Failure where
  stage : Stage
  currentDevice : Nat
structure Returned where
  objectId : Nat
  tensor : Admission.Tensor
  memory : Memory
  currentDevice : Nat
  launchDevice : Option Nat

/-- Runtime ports contain ordinary storage/copy metadata and Unit setup
tokens. They cannot supply source states or numerical kernel outputs. -/
structure InnerRuntime (r : Admission.Inner) (before : Memory) where
  empty : Except String (InnerPlace r)
  emptyCopy : Except String Unit
  prepare : Except String (InnerPrepared r before)
  setup : InnerPrepared r before → Nat → Except String Unit
  finish : (p : InnerPrepared r before) → Except String (InnerFinish r before p)

structure PrefixRuntime (r : Admission.Prefix) (before : Memory) where
  BS : Nat
  selected : CheckedPrefixNative.Selected BS
  empty : Except String (PrefixPlace r)
  prepare : Except String (PrefixPrepared r before)
  setup : PrefixPrepared r before → Nat → Except String Unit
  finish : (p : PrefixPrepared r before) → Except String (PrefixFinish r before p)

def runInner (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (rt : InnerRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) : Except Failure Returned :=
  if r.check then
    if r.nonempty then
      match rt.prepare with
      | .error _=>.error ⟨.prepare,caller⟩
      | .ok p=>if p.check then
          match rt.setup p r.query.device with
          | .error _=>.error ⟨.setup,caller⟩
          | .ok _=>match rt.finish p with
            | .error _=>.error ⟨.finish,caller⟩
            | .ok finish=>.ok ⟨finish.place.objectId,finish.place.tensor,
                ReturnedMemory.innerBytes I r before p finish scratch,caller,some r.query.device⟩
        else .error ⟨.pointers,caller⟩
    else
      let allocated:=match rt.empty with
        | .error _=>.error ⟨.empty,caller⟩
        | .ok place=>.ok ⟨place.objectId,place.tensor,ReturnedMemory.emptyBytes I r before,caller,none⟩
      if r.final.isSome then match rt.emptyCopy with
        | .error _=>.error ⟨.empty,caller⟩
        | .ok _=>allocated
      else allocated
  else .error ⟨.admission,caller⟩

def runPrefix (I : ScalarNative.Primitives) (r : Admission.Prefix) (before : Memory)
    (rt : PrefixRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) : Except Failure Returned :=
  if r.check then
    if r.nonempty then
      match rt.prepare with
      | .error _=>.error ⟨.prepare,caller⟩
      | .ok p=>if p.check then
          match rt.setup p r.input.device with
          | .error _=>.error ⟨.setup,caller⟩
          | .ok _=>match rt.finish p with
            | .error _=>.error ⟨.finish,caller⟩
            | .ok finish=>.ok ⟨finish.place.objectId,finish.place.tensor,
                ReturnedMemory.prefixBytes I r rt.BS before p finish scratch,caller,some r.input.device⟩
        else .error ⟨.pointers,caller⟩
    else match rt.empty with
      | .error _=>.error ⟨.empty,caller⟩
      | .ok place=>.ok ⟨place.objectId,place.tensor,before,caller,none⟩
  else .error ⟨.admission,caller⟩

inductive InnerSuccess (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (rt : InnerRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) : Returned → Prop
  | empty (checked : r.check=Bool.true) (empty : ¬r.nonempty) (place : InnerPlace r)
      (allocation : rt.empty=.ok place) (copy : r.final.isSome=Bool.true → rt.emptyCopy=.ok ()) :
      InnerSuccess I r before rt scratch caller
        ⟨place.objectId,place.tensor,ReturnedMemory.emptyBytes I r before,caller,none⟩
  | active (checked : r.check=Bool.true) (nonempty : r.nonempty) (p : InnerPrepared r before)
      (allocation : rt.prepare=.ok p) (pointers : p.check=Bool.true)
      (setup : rt.setup p r.query.device=.ok ()) (finish : InnerFinish r before p) (returned : rt.finish p=.ok finish) :
      InnerSuccess I r before rt scratch caller
        ⟨finish.place.objectId,finish.place.tensor,ReturnedMemory.innerBytes I r before p finish scratch,caller,some r.query.device⟩

inductive PrefixSuccess (I : ScalarNative.Primitives) (r : Admission.Prefix) (before : Memory)
    (rt : PrefixRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) : Returned → Prop
  | empty (checked : r.check=Bool.true) (empty : ¬r.nonempty) (place : PrefixPlace r) (allocation : rt.empty=.ok place) :
      PrefixSuccess I r before rt scratch caller ⟨place.objectId,place.tensor,before,caller,none⟩
  | active (checked : r.check=Bool.true) (nonempty : r.nonempty) (p : PrefixPrepared r before)
      (allocation : rt.prepare=.ok p) (pointers : p.check=Bool.true)
      (setup : rt.setup p r.input.device=.ok ()) (finish : PrefixFinish r before p) (returned : rt.finish p=.ok finish) :
      PrefixSuccess I r before rt scratch caller
        ⟨finish.place.objectId,finish.place.tensor,ReturnedMemory.prefixBytes I r rt.BS before p finish scratch,caller,some r.input.device⟩

def InnerSpec (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (rt : InnerRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) : Except Failure Returned→Prop
  | .error failure=>failure.currentDevice=caller
  | .ok result=>InnerSuccess I r before rt scratch caller result
def PrefixSpec (I : ScalarNative.Primitives) (r : Admission.Prefix) (before : Memory)
    (rt : PrefixRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) : Except Failure Returned→Prop
  | .error failure=>failure.currentDevice=caller
  | .ok result=>PrefixSuccess I r before rt scratch caller result

theorem inner_complete (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (rt : InnerRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) :
    InnerSpec I r before rt scratch caller (runInner I r before rt scratch caller) := by
  unfold runInner
  split
  · rename_i checked
    split
    · rename_i nonempty
      cases allocation : rt.prepare with
      | error reason=>rfl
      | ok p=>
        simp only
        split
        · rename_i pointers
          cases setup : rt.setup p r.query.device with
          | error reason=>rfl
          | ok token=>
            cases token
            cases finish : rt.finish p with
            | error reason=>rfl
            | ok f=>exact .active checked nonempty p allocation pointers setup f finish
        · rfl
    · rename_i empty
      simp only
      split
      · rename_i final
        cases copy : rt.emptyCopy with
        | error reason=>rfl
        | ok token=>
          cases token
          cases allocation : rt.empty with
          | error reason=>rfl
          | ok place=>exact .empty checked empty place allocation (fun _=>copy)
      · rename_i noFinal
        cases allocation : rt.empty with
        | error reason=>rfl
        | ok place=>exact .empty checked empty place allocation (fun h=>False.elim (noFinal h))
  · rfl

theorem prefix_complete (I : ScalarNative.Primitives) (r : Admission.Prefix) (before : Memory)
    (rt : PrefixRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat) :
    PrefixSpec I r before rt scratch caller (runPrefix I r before rt scratch caller) := by
  unfold runPrefix
  split
  · rename_i checked
    split
    · rename_i nonempty
      cases allocation : rt.prepare with
      | error reason=>rfl
      | ok p=>
        simp only
        split
        · rename_i pointers
          cases setup : rt.setup p r.input.device with
          | error reason=>rfl
          | ok token=>
            cases token
            cases finish : rt.finish p with
            | error reason=>rfl
            | ok f=>exact .active checked nonempty p allocation pointers setup f finish
        · rfl
    · rename_i empty
      cases allocation : rt.empty with
      | error reason=>rfl
      | ok place=>exact .empty checked empty place allocation
  · rfl

#axiomsClean inner_complete
#axiomsClean prefix_complete
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CheckedHost
