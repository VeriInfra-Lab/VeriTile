import bench.tritonbench_g.lightning_attention.Proofs.PairExecution
import bench.tritonbench_g.lightning_attention.Proofs.NativeResults
import bench.tritonbench_g.lightning_attention.Proofs.NativeSafety
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
noncomputable section
set_option maxHeartbeats 4000000

/-- A narrow source-schedule view of the existing materialized IO. Domain,
codec, observer, input/output channels and memory footprints are fixed; the
only field is an actual schedule of shared MaterializedInvocations. -/
structure Implementation (io : MaterializedProgramIO Word Word Native.Expr)
    (domain : io.Inputs → Prop) where
  schedule : io.Inputs → List (MaterializedInvocation Word Native.Expr)

def Implementation.Equivalent {io : MaterializedProgramIO Word Word Native.Expr}
    {domain : io.Inputs → Prop} (left right : Implementation io domain) : Prop :=
  ∀ xs, domain xs → ∀ initial, io.InputCells xs initial →
    (∀ bounds, io.InBounds bounds xs →
      MaterializedScheduleSafe bounds (left.schedule xs) initial ∧
        MaterializedScheduleSafe bounds (right.schedule xs) initial) ∧
    ∃ l r, runMaterializedSchedule (left.schedule xs) initial=some l ∧
      runMaterializedSchedule (right.schedule xs) initial=some r ∧
      (∀ o j, io.outputMask o xs j →
        mapCell (io.observe o j) (l io.outputs[o].buf (io.write o xs j)) =
          mapCell (io.observe o j) (r io.outputs[o].buf (io.write o xs j))) ∧
      io.Frame xs initial l ∧ io.Frame xs initial r

instance programSyntax {io : MaterializedProgramIO Word Word Native.Expr}
    {domain : io.Inputs → Prop} : Spec.ProgramSyntax (Implementation io domain) where
  Statement := Implementation io domain
  Signature := Unit
  signature := fun _ => ()
  body := fun p => [p]
  structural := some Implementation.Equivalent
  sameContext := Eq

namespace Forward

def original (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    Implementation (NativeIO.forward ops plan d) (fun _ => True) where
  schedule := (NativeIO.forward ops plan d).schedule

def candidate (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    Implementation (NativeIO.forward ops plan d) (fun _ => True) where
  schedule := fun _ => PairExecution.schedule ops plan .forward d

theorem run_eq (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (xs : (NativeIO.forward ops plan d).Inputs) (memory : Memory Word) :
    runMaterializedSchedule ((candidate ops plan d).schedule xs) memory =
      runMaterializedSchedule ((original ops plan d).schedule xs) memory :=
  PairExecution.stage_run ops plan .forward d _ memory

theorem safe_iff (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (xs : (NativeIO.forward ops plan d).Inputs) (bounds : RegionName → Nat) (memory : Memory Word) :
    MaterializedScheduleSafe bounds ((candidate ops plan d).schedule xs) memory ↔
      MaterializedScheduleSafe bounds ((original ops plan d).schedule xs) memory :=
  PairExecution.stage_safe ops plan .forward d _ bounds memory

/-- All original dimensions and words, including overlapping padded windows,
each CTA's actual materialization, every output cell and exterior frame. -/
theorem equivalent (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    Implementation.Equivalent (original ops plan d) (candidate ops plan d) := by
  intro xs _ initial binding
  obtain ⟨final,ran,outputs,frame⟩ := NativeResults.forward ops plan d xs initial binding
  refine ⟨?_,final,final,ran,(run_eq ops plan d xs initial).trans ran,?_,frame,frame⟩
  · intro bounds backed
    have safe := NativeSafety.forward ops plan d xs bounds backed initial
    exact ⟨safe,(safe_iff ops plan d xs bounds initial).mpr safe⟩
  · intro o j _
    rfl
end Forward

namespace Backward

def original (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    Implementation (NativeIO.backward ops plan d) (fun _ => True) where
  schedule := (NativeIO.backward ops plan d).schedule

def candidate (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    Implementation (NativeIO.backward ops plan d) (fun _ => True) where
  schedule := fun _ => PairExecution.schedule ops plan .intra d ++ PairExecution.schedule ops plan .inter d

theorem run_eq (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (xs : (NativeIO.backward ops plan d).Inputs) (memory : Memory Word) :
    runMaterializedSchedule ((candidate ops plan d).schedule xs) memory =
      runMaterializedSchedule ((original ops plan d).schedule xs) memory := by
  simp only [candidate,original,NativeIO.backward,NativeIO.schedule,PairExecution.schedule,
    runMaterializedSchedule_append,PairExecution.stage_run]

theorem safe_iff (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (xs : (NativeIO.backward ops plan d).Inputs) (bounds : RegionName → Nat) (memory : Memory Word) :
    MaterializedScheduleSafe bounds ((candidate ops plan d).schedule xs) memory ↔
      MaterializedScheduleSafe bounds ((original ops plan d).schedule xs) memory := by
  simp only [candidate,original,NativeIO.backward,NativeIO.schedule,PairExecution.schedule,
    materializedScheduleSafe_append,runMaterializedSchedule_append,
    PairExecution.stage_run,PairExecution.stage_safe]

/-- Complete intra grid followed by complete inter grid. Inter consumes the
actual predecessor gradients after materialization, on the original all-word
domain; no Host.Contract or finite-value restriction is added. -/
theorem equivalent (ops : Instructions) (plan : Plan) (d : Host.Dimensions) :
    Implementation.Equivalent (original ops plan d) (candidate ops plan d) := by
  intro xs _ initial binding
  obtain ⟨final,ran,outputs,frame⟩ := NativeResults.backward ops plan d xs initial binding
  refine ⟨?_,final,final,ran,(run_eq ops plan d xs initial).trans ran,?_,frame,frame⟩
  · intro bounds backed
    have safe := NativeSafety.backward ops plan d xs bounds backed initial
    exact ⟨safe,(safe_iff ops plan d xs bounds initial).mpr safe⟩
  · intro o j _
    rfl
end Backward

#axiomsClean Forward.equivalent
#axiomsClean Backward.equivalent
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairIO
