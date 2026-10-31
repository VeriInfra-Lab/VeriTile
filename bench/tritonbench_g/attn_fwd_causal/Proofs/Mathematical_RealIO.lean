import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealAccessSafety

/-! Real IO over the finite input windows actually visited by this program.
The number of visits may depend on the program ID; no fixed experiment shape
or context-sized stream truncates the source prefix. Input values are pinned
only where the source reads them. The existing FlatAlloc execution bridge
provides successful flat-memory execution, output readback and the full frame.
This is a per-program specification, with launch composition kept separate. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIO
open VeriTile.Triton RealLoads
noncomputable section

structure IO (BM BN BD : Nat) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  configuration : Nat → Nat → Config
  out : RegionName
  stage : Nat

def IO.at {BM BN BD : Nat} (io : IO BM BN BD) (s : BlockState) : Config :=
  io.configuration (s.pids 0) (s.pids 1)

def IO.regions {BM BN BD : Nat} (io : IO BM BN BD) (s : BlockState) : List RegionName :=
  let c := io.at s
  [c.Q,c.K,c.V,c.QS,c.KS,io.out]

def InjectiveOutput (c : Config) (BM BD : Nat) : Prop :=
  ∀ a b, RealEpilogue.active c BM BD a = Bool.true → RealEpilogue.active c BM BD b = Bool.true →
    RealEpilogue.address c BM BD a = RealEpilogue.address c BM BD b → a = b

structure Results {BM BN BD : Nat} (io : IO BM BN BD) (s t : BlockState)
    (ys : TileIndex [BM,BD] → ℝ) : Prop where
  output : ∀ i, RealEpilogue.active (io.at s) BM BD i = Bool.true →
    t.readMem io.out (RealEpilogue.address (io.at s) BM BD i) = ys i
  frame : ∀ r offset, (r ≠ io.out ∨ ∀ i, RealEpilogue.active (io.at s) BM BD i = Bool.true →
    offset ≠ RealEpilogue.address (io.at s) BM BD i) → t.mem r offset = s.mem r offset

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIO
