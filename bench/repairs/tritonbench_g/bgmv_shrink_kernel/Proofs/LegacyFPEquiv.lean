/- Historical padded/snapshot wrapper theorem, retained proof credit. -/
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.FPReadback
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.FPWorkset
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.NativeWords
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Preparation
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.HostProtocol

/-! Independent FP reader entry. Both source input branches retain their actual
FP16/BF16 product and complete typed reduction tensor, FP32 extension and
ordered accumulator addition, separate FP32 scaling, output conversion and
storage-specific atomic operation. No multiplication/addition is replaced by
FMA, no reduction is reassociated, and no atomic order is restricted.
The production output is FP32 under the user's default for an unspecified
role; both original FP16/BF16 input branches are retained. Formal generic
output types and additional FP16/BF16 compilation evidence do not claim
integer or FP64 output coverage. No new Python output-dtype guard is imposed.

`sourcePrograms` proves complete successful source executions, all payload
readbacks and frames. `exactEventCoverage` proves the single grid's 64-event
fiber for each active output cell. `Workset.Orders` represents the ordinary
runtime's actual arbitrary linearization separately for each output cell.
`floatingTarget` is its ordered local-primitive fold. Numerical deferral is
only instantiation of FPValues.Algebra/Primitives operations at the stated
types/layouts, including special values; it is never an output/execution law.
The ordinary tensor snapshot, allocation, scalar encoding and final copy
contracts remain explicit. Generic launch/PID compatibility is globally deferred.
-/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.FPEquiv
open VeriTile.Triton FP.Structural FPValues CallerOutput
noncomputable section
variable {α : Type} [Inhabited α]

abbrev sourcePrograms := @FPReadback.every_program
abbrev exactEventCoverage := Workset.fiber
abbrev floatingTarget := @FPValues.target
abbrev NativeAddresses := NativeWords.linked
abbrev sourceAtomicEvent := @FPWorkset.source_event
abbrev protocolFrame := @HostProtocol.frame
abbrev protocolIdentity := @HostProtocol.identity

def scratch (M : Algebra α) (P : Primitives α) (h : Call) (orders : Workset.Orders h)
    (data : Data α) (b n : Nat) : α :=
  if inside : b<h.batches ∧ n<h.N then
    floatingTarget M P h orders data ⟨b,inside.1⟩ ⟨n,inside.2⟩
  else data.output (Geometry.outputOffset h b n)

/-- Caller-byte readback of the complete event workset's actual per-cell fold.
This theorem composes the coordinate-preserving final tensor copy with that
fold. Its payload/source link is `sourcePrograms`, and exact event coverage is
`exactEventCoverage`; both are separately Lean-checked, not premises here. -/
theorem returned_equiv (M : Algebra α) (P : Primitives α) (h : Call)
    (orders : Workset.Orders h) (data : Data α) (v : View)
    (guard : Shared.Admitted v.original) (encode : Bool → α → Fin v.width → BitVec 8)
    (before : Bytes) :
    let result := returned h v encode data.indices (scratch M P h orders data) before
    result.result=() ∧ result.view=v ∧
    (∀ i : Coord v,(active : Active h v data.indices i) → ∀ byte,
      result.memory v.original.owner (address v i byte)=
        encode v.original.negative
          (floatingTarget M P h orders data ⟨i.1.val,active.1⟩ ⟨i.2.val,active.2.1⟩) byte) ∧
    (∀ owner ptr,(¬∃ i : Coord v,∃ byte : Fin v.width,
      Active h v data.indices i ∧ owner=v.original.owner ∧ ptr=address v i byte) →
      result.memory owner ptr=before owner ptr) := by
  refine ⟨rfl,rfl,?_,?_⟩
  · intro i active byte
    change copy h v encode data.indices (scratch M P h orders data) before v.original.owner (address v i byte)=_
    rw [copy_byte h v encode guard data.indices _ before i active byte]
    unfold scratch
    rw [dif_pos ⟨active.1,active.2.1⟩]
  · exact CallerOutput.frame h v encode data.indices _ before

#axiomsClean returned_equiv

def runtimeScratch (M : Algebra α) (P : Primitives α) (h : Call) (orders : Workset.Orders h)
    (data : Data α) (ready : Preparation.Prepared α h data) (run : FPWorkset.Run M P h orders data)
    (b n : Nat) : α :=
  if inside : b<h.batches ∧ n<h.N then
    toFloat h.outputType ((FPWorkset.physical M P h orders data ready run
      (Preparation.outputAddress ready (⟨b,inside.1⟩,⟨n,inside.2⟩))).read h.outputType.toTileDType)
  else data.output (Geometry.outputOffset h b n)

/-- Complete source-workset/physical-readback/caller-copy composition. The
runtime supplies only local atomic linearization traces, each derived update
being linked to full source execution by `sourceAtomicEvent`. No folded
scratch result, successful source run or final output equality is a premise. -/
theorem returned_from_runtime (M : Algebra α) (P : Primitives α) (h : Call)
    (orders : Workset.Orders h) (data : Data α) (ready : Preparation.Prepared α h data)
    (valid : ∀ b,b<h.batches → data.indices b=-1 ∨
      (0≤data.indices b ∧ (data.indices b).toNat<h.adapters))
    (run : FPWorkset.Run M P h orders data) (v : View)
    (guard : Shared.Admitted v.original) (encode : Bool → α → Fin v.width → BitVec 8)
    (before : Bytes) :
    FPWorkset.SourceWorkset M P h data ∧
    (let result := returned h v encode data.indices
      (runtimeScratch M P h orders data ready run) before
    result.result=() ∧ result.view=v ∧
    (∀ i : Coord v,(active : Active h v data.indices i) → ∀ byte,
      result.memory v.original.owner (address v i byte)=
        encode v.original.negative
          (floatingTarget M P h orders data ⟨i.1.val,active.1⟩ ⟨i.2.val,active.2.1⟩) byte) ∧
    (∀ owner ptr,(¬∃ i : Coord v,∃ byte : Fin v.width,
      Active h v data.indices i ∧ owner=v.original.owner ∧ ptr=address v i byte) →
      result.memory owner ptr=before owner ptr)) := by
  constructor
  · exact FPWorkset.source_workset M P h data (fun b hb =>
      (valid b hb).imp_right And.left)
  refine ⟨rfl,rfl,?_,?_⟩
  · intro i active byte
    change copy h v encode data.indices _ before v.original.owner (address v i byte)=_
    rw [copy_byte h v encode guard data.indices _ before i active byte]
    unfold runtimeScratch
    rw [dif_pos ⟨active.1,active.2.1⟩]
    rw [FPWorkset.physical_readback M P h orders data ready run _ _ active.2.2,FPWorkset.decode]
  · exact CallerOutput.frame h v encode data.indices _ before

#axiomsClean returned_from_runtime
-- This is a structural Algebra/Primitives theorem, not a ProfiledRewrite
-- derivation accepted by #print_fp_assumptions. Its explicit local ports are
-- recorded in the frozen numerical ledger; #axiomsClean audits its proof.
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.FPEquiv
