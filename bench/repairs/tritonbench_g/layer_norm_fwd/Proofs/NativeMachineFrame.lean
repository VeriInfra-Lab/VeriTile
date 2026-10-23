import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineExecution

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

def Outside {α : Type} (c : Config α block) (row : Fin c.M) (i : Fin (layout c.allocations).capacity) : Prop :=
  ∀ region offset, Contract.written c.flags c.regions c.strides c.M c.N row.val region offset →
    (layout c.allocations).address (region,offset)≠i.val

theorem store_frame {α : Type} (c : Config α block) (row : Fin c.M) (col : Fin block)
    (role : Role) (hw : NativeMachineMemory.writeRole role) (physical : Memory c) (value : Cell α)
    (i : Fin (layout c.allocations).capacity) (hi : Outside c row i) :
    NativeMachineMemory.store c.allocations c.regions c.strides c.M c.N row.val col.val c.flags role physical value i=physical i := by
  rw [NativeMachineMemory.store_eq_physical c.flags c.regions c.strides c.M c.N block
    c.allocations c.accepted c.storage row.val col.val row.isLt col.isLt role physical value]
  by_cases h : active c.flags role ∧ NativeAccess.laneActive c.N col.val role
  · rw [if_pos h]
    have written : Contract.written c.flags c.regions c.strides c.M c.N row.val
        (region c.regions role) (offset c.strides c.M c.N row.val col.val role) := by
      cases role <;> simp_all [NativeMachineMemory.writeRole,active,NativeAccess.laneActive,region,offset,
        Contract.written,Contract.inRow_of_col]
    exact if_neg (Ne.symm (hi _ _ written))
  · rw [if_neg h]

theorem scatter_frame {α : Type} (c : Config α block) (row : Fin c.M)
    (role : Role) (hw : NativeMachineMemory.writeRole role) (values : Fin block → Cell α) (physical : Memory c)
    (i : Fin (layout c.allocations).capacity) (hi : Outside c row i) :
    NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags role values physical i=physical i := by
  have hseq : ∀ (indices : List (TileIndex [block])) (p : Memory c),
      (indices.foldl (fun t j => NativeMachineMemory.store c.allocations c.regions c.strides
        c.M c.N row.val j.1.val c.flags role t (values j.1)) p) i=p i := by
    intro indices
    induction indices with
    | nil => intro p; rfl
    | cons j js ih =>
      intro p
      rw [List.foldl_cons,ih]
      exact store_frame c row j.1 role hw p (values j.1) i hi
  exact hseq _ physical

theorem mask_frame {α : Type} (c : Config α block) (row : Fin c.M) (second : Bool)
    (values : NativeSchedule.StreamRegisters α block) (physical : Memory c)
    (i : Fin (layout c.allocations).capacity) (hi : Outside c row i) :
    storeMask c row second values physical i=physical i := by
  unfold storeMask
  split_ifs
  · exact scatter_frame c row (maskRole second) (by cases second <;> trivial) _ physical i hi
  · rfl

theorem input_frame {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M) (physical : Memory c)
    (i : Fin (layout c.allocations).capacity) (hi : Outside c row i) :
    (inputStage c row physical).memory i=physical i := by
  let first := stream c row Bool.false physical
  let t := storeMask c row Bool.false first physical
  let second := stream c row Bool.true t
  let u := if c.flags.x1 then storeMask c row Bool.true second t else t
  have ht : t i=physical i := mask_frame c row Bool.false first physical i hi
  have hu : u i=physical i := by
    dsimp only [u]
    split_ifs
    · rw [mask_frame c row Bool.true second t i hi,ht]
    · exact ht
  change (if c.flags.residualOut then NativeMachineMemory.scatter c.allocations c.regions c.strides
    c.M c.N row.val block c.flags .residualOut
      (fun j => Cell.mk .real (c.ops.cast32 (((inputStage c row physical).value j).eval c.ops))) u else u) i=physical i
  split_ifs
  · rw [scatter_frame c row .residualOut trivial _ u i hi,hu]
  · exact hu

theorem normalization_frame {α : Type} (c : Config α block) (row : Fin c.M) (input : InputRegisters c)
    (i : Fin (layout c.allocations).capacity) (hi : Outside c row i) :
    (normalizationStage c row input).memory i=input.memory i := by
  change NativeMachineMemory.store c.allocations c.regions c.strides c.M c.N row.val 0 c.flags .rstd
    (if c.flags.rms then input.memory else NativeMachineMemory.store c.allocations c.regions c.strides
      c.M c.N row.val 0 c.flags .mean input.memory
      (.mk .real (c.ops.cast32 (((normalizationStage c row input).mean 0).eval c.ops))))
    (.mk .real (c.ops.cast32 (((normalizationStage c row input).rstd 0).eval c.ops))) i=input.memory i
  rw [store_frame c row ⟨0,c.accepted.blockPositive⟩ .rstd trivial _ _ i hi]
  split_ifs
  · rfl
  · exact store_frame c row ⟨0,c.accepted.blockPositive⟩ .mean trivial _ _ i hi

theorem affine_frame {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M) (input : NormalRegisters c)
    (i : Fin (layout c.allocations).capacity) (hi : Outside c row i) :
    affineStage c row input i=input.memory i := by
  dsimp only [affineStage]
  by_cases hw : c.flags.w1=Bool.true
  · rw [if_pos hw,scatter_frame c row .y1 trivial _ _ i hi]
    exact scatter_frame c row .y trivial _ _ i hi
  · rw [if_neg hw]
    exact scatter_frame c row .y trivial _ _ i hi

/-- Every physical cell outside this row's actual output addresses is
preserved, including cells not named by any logical tensor view. -/
theorem execute_frame {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M) (physical : Memory c)
    (i : Fin (layout c.allocations).capacity) (hi : Outside c row i) :
    execute c row physical i=physical i := by
  rw [execute,affine_frame c row _ i hi,normalization_frame c row _ i hi,input_frame c row physical i hi]

#axiomsClean store_frame
#axiomsClean scatter_frame
#axiomsClean mask_frame
#axiomsClean input_frame
#axiomsClean normalization_frame
#axiomsClean affine_frame
#axiomsClean execute_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
