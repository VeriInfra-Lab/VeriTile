import bench.tritonbench_g.rotary_emb.Kernels
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.rotary_emb.Proofs.ProjectionLists

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def storageCarrier : Storage → TileDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .f8e4 => .f8e4
  | .f8e5 => .f8e5

def storedCarrier : ComputeStmt → Option TileDType
  | .store dtype _ _ _ _ => some dtype
  | _ => none

def promotedStorage (a b : Storage) : Storage :=
  if a=.fp64 ∨ b=.fp64 then .fp64 else
  if a=.fp32 ∨ b=.fp32 then .fp32 else
  if a=.fp16 ∨ b=.fp16 then .fp16 else
  if a=.bf16 ∧ b=.bf16 then .bf16 else
  if a=.bf16 ∨ b=.bf16 then .fp32 else
  if a=b then a else .fp16

/-- Raw operation carrier, before relating the two representations of fp32.
Promoting only narrow operands can produce `.fp32`; an explicit source fp32
load uses `.real` together with `ComputeDType.fp32`. -/
def rawAssignedCarrier : ComputeStmt → Option TileDType
  | .assign dtype _ _ _ => some dtype
  | _ => none

/-- Canonical carrier for the logical source storage type. Both `.fp32` and
the compute-tagged `.real` representation implement source fp32 arithmetic.
The separate precision check below distinguishes the fp64 `.real` channel. -/
def assignedCarrier (st : ComputeStmt) : Option TileDType :=
  match rawAssignedCarrier st with
  | some .fp32 => some .real
  | other => other

def assignedPrecision : ComputeStmt → Option ComputeDType
  | .assign _ _ _ (.compute (.alg precision _)) => some precision
  | .assign .fp32 _ _ (.alg _) => some .fp32
  | _ => none

def storagePrecision : Storage → Option ComputeDType
  | .fp32 => some .fp32
  | .fp64 => some .fp64
  | _ => none

local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

theorem template_shape (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).surfaceBody.length = 34 := by
  cases data <;> cases cos <;> cases sin <;> rfl

theorem template_stores (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    let body := (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).surfaceBody
    storedCarrier body[18]! = some (storageCarrier data) ∧
    storedCarrier body[19]! = some (storageCarrier data) ∧
    storedCarrier body[32]! = some (storageCarrier data) ∧
    storedCarrier body[33]! = some (storageCarrier data) := by
  cases data <;> cases cos <;> cases sin <;> exact ⟨rfl,rfl,rfl,rfl⟩

/-- The two products promote independently before the final sum/difference.
This also checks the fp32/fp64 compute tag, whose erased tile carrier is shared. -/
theorem template_result_precision (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    let body := (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd
      ssb ssd L HQ HK BH BS D).surfaceBody
    let dtype := promotedStorage (promotedStorage data cos) (promotedStorage data sin)
    assignedCarrier body[16]! = some (storageCarrier dtype) ∧
    assignedCarrier body[17]! = some (storageCarrier dtype) ∧
    assignedPrecision body[16]! = storagePrecision dtype ∧
    assignedPrecision body[17]! = storagePrecision dtype := by
  cases data <;> cases cos <;> cases sin <;> exact ⟨rfl,rfl,rfl,rfl⟩

/-- Source fp32 can arise from promotion without an explicit compute wrapper.
Keep the raw carrier visible so the canonical check cannot conceal which AST
representation was actually emitted. Both even and odd expressions agree. -/
theorem promoted_fp32_raw_carrier (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    let body := (template .bf16 .fp16 .f8e4 Q K Cos Sin sqb sqh sqd skb skh skd
      scb scd ssb ssd L HQ HK BH BS D).surfaceBody
    rawAssignedCarrier body[16]! = some .fp32 ∧
    rawAssignedCarrier body[17]! = some .fp32 := ⟨rfl, rfl⟩

theorem original_shape (q k cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (original q k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).surfaceBody.length = 34 := by
  change ((template q cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd
    ssb ssd L HQ HK BH BS D).surfaceBody.take 20 ++
    (template k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd
      ssb ssd L HQ HK BH BS D).surfaceBody.drop 20).length = 34
  simp only [List.length_append, List.length_take, List.length_drop, template_shape] <;> decide

private theorem template_projectable (data cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D).toAlgorithm? = .ok
    (template data cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D).toAlgKernel := by
  cases data <;> cases cos <;> cases sin <;> rfl

private theorem projected_roles (ins outs : List RegionName) (body : List ComputeStmt)
    (h : (ComputeKernel.mk ins outs body).toAlgorithm? =
      .ok (ComputeKernel.mk ins outs body).toAlgKernel) :
    (ComputeKernel.mk ins outs body).inputs = ins ∧
      (ComputeKernel.mk ins outs body).outputs = outs := by
  cases hb : ComputeStmt.listToAlgorithm? body <;>
    simp_all [ComputeKernel.inputs, ComputeKernel.outputs,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_mk]

theorem io_roles (q k cos sin : Storage) (v : Bool) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (kernel q k cos sin v Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).inputs = [Q,K,Cos,Sin] ∧
    (kernel q k cos sin v Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D).outputs = [Q,K] := by
  let a := original q k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
    L HQ HK BH BS D
  have ha : a.toAlgorithm? = .ok a.toAlgKernel := by
    exact ProjectionLists.joined_supported _ _ _ _ 20
      (template_projectable q cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd
        ssb ssd L HQ HK BH BS D)
      (template_projectable k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd
        ssb ssd L HQ HK BH BS D)
  have hroles : a.inputs = [Q,K,Cos,Sin] ∧ a.outputs = [Q,K] :=
    projected_roles _ _ _ ha
  cases v
  · exact hroles
  · let replacement : ComputeStmt := .assign .nat [BH] "cur_head_range"
      (.alg (.add .nat .scalarL
        (.mul .nat .nil (.constNat BH) (.ref .nat [] "cur_head_index")) (.arange BH)))
    have hp := ProjectionLists.replace_supported a a.inputs a.outputs 2 replacement
      (.assign .nat [BH] "cur_head_range" (.add .nat .scalarL
        (.mul .nat .nil (.constNat BH) (.ref .nat [] "cur_head_index")) (.arange BH))) ha rfl
    have hr := projected_roles _ _ _ hp
    exact ⟨hr.1.trans hroles.1, hr.2.trans hroles.2⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource
