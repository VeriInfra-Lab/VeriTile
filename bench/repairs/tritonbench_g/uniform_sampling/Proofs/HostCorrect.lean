import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.LegacySource
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.HostContract
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.RealPublic

namespace VeriTile.Bench.Repairs.UniformSampling.Correct
open VeriTile.Triton VeriTile.Torch
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

specification checked_returned (api : API ℝ Store Byte) (denote : Float64Bits→ℝ) (scalars : Scalars)
    (output : TensorMetadata) (generator : GeneratorArgument)
    (contract : InputContract output generator lowerIsFloat upperIsFloat)
    (v : View n) (_binding : TensorBinding output v) (scratch : Scratch n)
    (defaults : Nat→Nat) (which : Nat)
    (_selection : which=selectedGenerator defaults output.device.index generator)
    (lib : Host.Runtime) (stops : Host.Stops) (s : Host.State Store) (out : Nat) (hn:0<n)
    (ok:((Host.entry api (model denote) scalars v scratch which lib stops).run s).result=.ok out) :
    ∃ r g, guarded output generator lowerIsFloat upperIsFloat
        ((Host.entry api (model denote) scalars v scratch which lib stops).run s)=
          .ok ((Host.entry api (model denote) scalars v scratch which lib stops).run s) ∧
      reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g) ∧
      out=v.handle ∧
      (∀ i,i<n→logical api (model denote)
        ((Host.entry api (model denote) scalars v scratch which lib stops).run s).state.memory v i=
          Spec.target (denote scalars.lower) (denote scalars.upper) r.seed r.start (block n) i) ∧
      ((Host.entry api (model denote) scalars v scratch which lib stops).run s).state.generators=
        Function.update s.generators which g ∧
      ((Host.entry api (model denote) scalars v scratch which lib stops).run s).state.device=s.device := by
  obtain ⟨r,g,hr,ho,hv,hg,hd⟩:=returned api denote scalars v scratch which lib stops s out hn ok
  exact ⟨r,g,guarded_eq contract _,hr,ho,hv,hg,hd⟩

abbrev empty := @checked_empty
abbrev all_outcomes := @checked_all_outcomes
#axiomsClean checked_returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Correct
