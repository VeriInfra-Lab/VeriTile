import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.ProgramPair
import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Interface
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.IOPair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

/-- The complete existing IO success/output/frame proof supplies the left run.
The actual complete-source equality transports that run and trace safety to the
candidate, preserving the same final state and both exterior memory frames. -/
theorem equivalent (cfg : FP.Config) (r : Request) (QR CR SR : Nat) (backward : Bool)
    (row group : Nat) (launch : r.Launched row group)
    (strideEq : row * QR = row * (r.heads * r.dimension))
    (cosineEq : CR = r.half) (sineEq : SR = r.half) :
    (io cfg r QR CR SR backward row group).Equivalent
      (variant cfg r QR CR SR backward row group) := by
  intro values _ bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := Interface.implements
    (Numerics.operations (Numerics.softwarePrimitives cfg)) r QR CR SR backward row group
    launch strideEq cosineEq sineEq values trivial bounds hb s hin
  have same := ProgramPair.io_execution (Numerics.operations (Numerics.softwarePrimitives cfg))
    r QR CR SR backward row group values s
  refine ⟨safe,(same.2 bounds).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · exact same.1.symm.trans ran
  · intro o j active
    rfl
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.IOPair
