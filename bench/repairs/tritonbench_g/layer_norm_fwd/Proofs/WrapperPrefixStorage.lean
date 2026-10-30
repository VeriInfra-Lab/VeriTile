import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorage

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

theorem persists_validate (store : TensorStore S Descriptor Value) (q : Request) :
    Persists (world:=world) store (validate q) := by
  unfold validate
  cases hr : q.residual
  all_goals
    first
    | apply persists_bind store _ _ (persists_pure store _)
    | apply persists_bind store _ _ (persists_checked store _ _)
    intro residualDtype
    apply persists_bind store _ _ (persists_checked store _ _)
    rintro ⟨M,N⟩
    apply persists_bind store _ _ (persists_checkStride store .x q.x)
    intro _
    apply persists_bind store _ _ (persists_forOption store _ _ (fun tensor =>
      persists_bind store _ _ (persists_checkStride store .residual tensor)
        (fun _ => persists_checkShape store .residual tensor [M,N])))
    intro _
    apply persists_bind store _ _ (persists_checkShape store .weight q.weight [N])
    intro _
    apply persists_bind store _ _ (persists_checkStride store .weight q.weight)
    intro _
    apply persists_bind store _ _ (persists_forOption store _ _ (fun tensor =>
      persists_bind store _ _ (persists_checkStride store .bias tensor)
        (fun _ => persists_checkShape store .bias tensor [N])))
    intro _
    apply persists_bind store _ _ (persists_forOption store _ _ (fun tensor =>
      persists_bind store _ _ (persists_checkShape store .x1 tensor q.x.shape) (fun _ =>
        persists_bind store _ _ (persists_check store _ _ _) (fun _ => persists_checkStride store .x1 tensor))))
    intro _
    apply persists_bind store _ _ (persists_forOption store _ _ (fun tensor =>
      persists_bind store _ _ (persists_checkShape store .weight1 tensor [N])
        (fun _ => persists_checkStride store .weight1 tensor)))
    intro _
    apply persists_bind store _ _ (persists_forOption store _ _ (fun tensor =>
      persists_bind store _ _ (persists_checkShape store .bias1 tensor [N])
        (fun _ => persists_checkStride store .bias1 tensor)))
    intro _
    apply persists_bind store _ _ (persists_forOption store _ _ (fun tensor =>
      persists_bind store _ _ (persists_checked store _ _) (fun _ =>
        persists_bind store _ _ (persists_check store _ _ _) (fun _ => persists_checkShape store .rowscale tensor [M]))))
    intro _
    exact persists_pure store _

theorem persists_allocateOptional (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (role : Role) (condition : Bool) (spec : Allocation) (assertStride : Bool) :
    Persists store (allocateOptional api role condition spec assertStride) := by
  unfold allocateOptional
  cases condition
  · exact persists_pure store none
  · apply persists_bind store _ _ (persists_call store _ _ (hc.allocate_preserves role spec))
    intro tensor
    cases assertStride
    · exact persists_bind store _ _ (persists_pure store _) (fun _ => persists_pure store (some tensor))
    · exact persists_bind store _ _ (persists_checkStride store role tensor) (fun _ => persists_pure store (some tensor))

theorem persists_allocateOutputs (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) (v : Validated) (input : Inputs) :
    Persists store (allocateOutputs api q v input) := by
  unfold allocateOutputs
  apply persists_bind store _ _ (persists_call store _ _ (hc.allocate_preserves _ _))
  intro y
  apply persists_bind store _ _ (persists_checkStride store .y y)
  intro _
  apply persists_bind store _ _ (persists_allocateOptional store api hc _ _ _ _)
  intro y1
  apply persists_bind store _ _ (persists_allocateOptional store api hc _ _ _ _)
  intro residualOut
  apply persists_bind store _ _ (persists_allocateOptional store api hc _ _ _ _)
  intro mean
  apply persists_bind store _ _ (persists_call store _ _ (hc.allocate_preserves _ _))
  intro rstd
  apply persists_bind store _ _ (persists_allocateOptional store api hc _ _ _ _)
  intro seeds
  apply persists_bind store _ _ (persists_allocateOptional store api hc _ _ _ _)
  intro mask
  exact persists_pure store _

theorem persists_width (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (v : Validated) (input : Inputs) : Persists store (width api v input) := by
  unfold width
  apply persists_bind store _ _ (persists_checked store _ _)
  intro bytes
  apply persists_bind store _ _ (persists_call store _ _ (hc.power_preserves _))
  intro next
  apply persists_bind store _ _ (persists_check store _ _ _)
  intro _
  exact persists_pure store _

/-- Every pre-launch outcome preserves all previously live tensor descriptors
and cells. New resolve/allocation effects remain in the actual prefix state. -/
theorem persists_prepare (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) : Persists store (prepare api q) :=
  persists_bind store _ _ (persists_validate store q) (fun validated =>
    persists_bind store _ _ (persists_resolveInputs store api hc q) (fun inputs =>
      persists_bind store _ _ (persists_allocateOutputs store api hc q validated inputs) (fun _ =>
        persists_bind store _ _ (persists_width store api hc validated inputs) (fun _ => persists_pure store _))))

#axiomsClean persists_validate
#axiomsClean persists_allocateOptional
#axiomsClean persists_allocateOutputs
#axiomsClean persists_width
#axiomsClean persists_prepare
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
