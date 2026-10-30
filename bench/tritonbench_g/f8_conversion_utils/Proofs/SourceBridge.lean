import bench.tritonbench_g.f8_conversion_utils.Proofs.SourceReadback
import bench.tritonbench_g.f8_conversion_utils.Proofs.Bytes
import VeriTile.Triton.Float.ExecutionProfile

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source
open VeriTile.Triton FP.Structural
open Bytes
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def inputCell {α : Type} (p : Profile) (value : α) : Cell α :=
  .mk p.input.toTileDType (ofFloat p.input value)

def outputValue {α : Type} [Inhabited α] (p : Profile) (cell : Cell α) : α :=
  toFloat p.output (cell.read p.output.toTileDType)

/-- Only a single scalar typed load/conversion/store ABI is related to words.
In the FP32 arm this explicitly contains M.fp32Load before the destination
cast; neither its effect nor any roundtrip/losslessness law is assumed. -/
structure ScalarBinding {α : Type} (M : Algebra α) (p : Profile)
    (physical : Word p.inputWidth → Word p.outputWidth) where
  loadWord : Word p.inputWidth → α
  storeWord : α → Word p.outputWidth
  conversion : ∀ word,storeWord (cast (M.withDefaultPrecision .fp32) p (loadWord word))=physical word

/-- The actual original whole source executes under the production profile.
Its returned typed cells, including the duplicate decode store, have the same
word observation as the configured local native conversion. -/
theorem native_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (physical : Word p.inputWidth → Word p.outputWidth) (binding : ScalarBinding M p physical)
    (Y X : RegionName) (N B : Nat) (s : State α) (words : Nat → Word p.inputWidth)
    (input : ∀ i,s.mem X i=inputCell p (binding.loadWord (words i))) :
    ∃ final,FP.Structural.exec (M.withDefaultPrecision .fp32) (kernel p Y X N B) s=some final ∧
      ∀ j : Fin B,s.pids 0*B+j.val<N →
        binding.storeWord (outputValue p (final.mem Y (s.pids 0*B+j.val)))=physical (words (s.pids 0*B+j.val)) := by
  obtain ⟨final,executed,readback⟩ := executes_readback (M.withDefaultPrecision .fp32) p Y X N B s
  refine ⟨final,executed,?_⟩
  intro j active
  rw [readback j active,input]
  have oneScalar := binding.conversion (words (s.pids 0*B+j.val))
  cases p <;> simpa [outputValue,inputCell,Profile.input,Profile.output,ofFloat,toFloat,Cell.read] using oneScalar

#axiomsClean native_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source
