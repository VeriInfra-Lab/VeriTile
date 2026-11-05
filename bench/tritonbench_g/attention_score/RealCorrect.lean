import bench.tritonbench_g.attention_score.Proofs.LogicalReader
import bench.tritonbench_g.attention_score.Proofs.RealIOProof
import bench.tritonbench_g.attention_score.Proofs.Reused.CurrentMachine

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Correct
open VeriTile Triton Reused Geometry Physical
open scoped BigOperators VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Exact shared source, full grid and flat Q/K/M channels. Out is an input
only for the existing empty dispatch, where the allocator supplies zero. -/
abbrev kernelIO := LogicalIO.io
def distance (row key : Nat) (offset : Int) : Int := (row : Int) - key + offset

/-- The complement arm keeps distance >= size, rather than negating both
bounds of the regular window. Negative offsets and sizes retain their meaning. -/
def windowAllows (window complement : Bool) (offset size : Int) (row key : Nat) : Prop :=
  window = Bool.false ∨ if complement then size ≤ distance row key offset
    else 0 ≤ distance row key offset ∧ distance row key offset < size

instance (w c : Bool) (offset size : Int) (r k : Nat) :
    Decidable (windowAllows w c offset size r k) := by
  unfold windowAllows
  infer_instance

def scoreSum (c : Shape) (xs : LogicalIO.Inputs c) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : ℝ :=
  ∑ i : Fin c.N, if windowAllows window complement offset size i.val j.val then
    pow2 ((∑ d : Fin c.D, xs.query z h i d * xs.key z (LogicalIO.keyHead c h) j d) *
      (scale*1.4426950408889634) - xs.normalizer z h i)
  else 0

/-- Every output equals the independent N-term windowed score sum. The shared
IO judgment includes actual source access safety and the complete exterior
frame. The original machine-range and tile-formation domain remains explicit;
its physical endpoint is retained in Proofs/Reused/RealPhysical. -/
specification kernel_correct (c : Shape) (abi : ABI c)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (_domain : Whole002.Representation.Domain c abi offset size window)
    (_tile : c.B*c.D≤1048576) :
    Spec.Real (kernelIO c offset size scale window complement ⊨{LogicalIO.initialOutput c}
      fun xs z h key => scoreSum c xs offset size scale window complement z h key) :=
  LogicalReader.correct c (fun hn ho => by
    have hk := (_domain.launch hn ho).paddedKey
    change c.programs*c.B≤2147483647 at hk
    omega) offset size scale window complement _ rfl

#specNonCircular scoreSum avoiding [kernel,realExecute,Triton.exec]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Correct
