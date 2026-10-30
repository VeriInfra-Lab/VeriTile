import bench.tritonbench_g.chunk_retention.Sources.KernelStaging
import bench.tritonbench_g.chunk_retention.Sources.ForwardHReadback
import bench.tritonbench_g.chunk_retention.Sources.Source

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative
open VeriTile Triton FP.Structural KernelModel KernelStaging
noncomputable section
variable {α : Type} [Inhabited α]

def inputs (ops : Ops α) (literal : ℝ → α) (c : Shape)
    (m : RegionName → Nat → Cell (Term α)) (seed terminal : Bool) : Inputs α where
  q := fun h t e => lower ops literal ((m "q" (h*(c.T*c.K)+t*c.K+e)).read .real)
  k := fun h t e => lower ops literal ((m "k" (h*(c.T*c.K)+t*c.K+e)).read .real)
  v := fun h t j => lower ops literal ((m "v" (h*(c.T*c.V)+t*c.V+j)).read .real)
  seed := fun h e j => if seed then lower ops literal ((m "h0" (h*c.K*c.V+e*c.V+j)).read .real) else ops.zero
  incoming := fun h t j => lower ops literal ((m "incoming" (h*(c.T*c.V)+t*c.V+j)).read .real)
  finalIncoming := fun h e j => if terminal then lower ops literal ((m "terminal" (h*c.K*c.V+e*c.V+j)).read .real) else ops.zero

theorem rate_eq (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (head : Nat) :
    lower ops literal (KernelModel.rate (algebra ops.zero) c head)=logRate ops c head := by
  simp [KernelModel.rate,algebra,lower,logRate,laws.one,laws.negativeFive]

theorem decay_eq (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (head : Nat) (n : Int) :
    lower ops literal (KernelModel.decay (algebra ops.zero) c head n)=decay ops c head n := by
  change ops.exp2 (ops.mul (ops.integer n)
    (lower ops literal (KernelModel.rate (algebra ops.zero) c head)))=_
  rw [rate_eq ops literal laws]
  rfl

theorem lower_ite (ops : Ops α) (literal : ℝ→α) (p : Prop) [Decidable p]
    (a b : Term α) : lower ops literal (if p then a else b)=
      if p then lower ops literal a else lower ops literal b := by split <;> rfl

#specNonCircular inputs avoiding [run,FP.Structural.exec]
#axiomsClean rate_eq
#axiomsClean decay_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative
