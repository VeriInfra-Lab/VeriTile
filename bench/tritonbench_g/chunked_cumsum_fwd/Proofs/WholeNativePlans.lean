import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan0
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan1
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan2
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan3
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan4
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan5
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan6
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlan7

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.NativePlans
open Backend
set_option maxHeartbeats 5000000
set_option maxRecDepth 100000

def plan_1_1 : PrefixPlan 1 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_1_128 : PrefixPlan 1 128 where
  tree := fun _ => NativePlan5.row
  leaves := fun _ => NativePlan5.prefix_correct

def plan_1_16 : PrefixPlan 1 16 where
  tree := fun _ => NativePlan2.row
  leaves := fun _ => NativePlan2.prefix_correct

def plan_1_256 : PrefixPlan 1 256 where
  tree := fun _ => NativePlan7.row
  leaves := fun _ => NativePlan7.prefix_correct

def plan_1_4 : PrefixPlan 1 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_1_64 : PrefixPlan 1 64 where
  tree := fun _ => NativePlan3.row
  leaves := fun _ => NativePlan3.prefix_correct

def plan_16_1 : PrefixPlan 16 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_16_128 : PrefixPlan 16 128 where
  tree := fun _ => NativePlan6.row
  leaves := fun _ => NativePlan6.prefix_correct

def plan_16_16 : PrefixPlan 16 16 where
  tree := fun _ => NativePlan2.row
  leaves := fun _ => NativePlan2.prefix_correct

def plan_16_256 : PrefixPlan 16 256 where
  tree := fun _ => NativePlan7.row
  leaves := fun _ => NativePlan7.prefix_correct

def plan_16_4 : PrefixPlan 16 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_16_64 : PrefixPlan 16 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

def plan_2_1 : PrefixPlan 2 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_2_128 : PrefixPlan 2 128 where
  tree := fun _ => NativePlan6.row
  leaves := fun _ => NativePlan6.prefix_correct

def plan_2_16 : PrefixPlan 2 16 where
  tree := fun _ => NativePlan2.row
  leaves := fun _ => NativePlan2.prefix_correct

def plan_2_256 : PrefixPlan 2 256 where
  tree := fun _ => NativePlan7.row
  leaves := fun _ => NativePlan7.prefix_correct

def plan_2_4 : PrefixPlan 2 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_2_64 : PrefixPlan 2 64 where
  tree := fun _ => NativePlan3.row
  leaves := fun _ => NativePlan3.prefix_correct

def plan_32_1 : PrefixPlan 32 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_32_128 : PrefixPlan 32 128 where
  tree := fun _ => NativePlan6.row
  leaves := fun _ => NativePlan6.prefix_correct

def plan_32_16 : PrefixPlan 32 16 where
  tree := fun _ => NativePlan2.row
  leaves := fun _ => NativePlan2.prefix_correct

def plan_32_256 : PrefixPlan 32 256 where
  tree := fun _ => NativePlan7.row
  leaves := fun _ => NativePlan7.prefix_correct

def plan_32_4 : PrefixPlan 32 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_32_64 : PrefixPlan 32 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

def plan_4_1 : PrefixPlan 4 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_4_128 : PrefixPlan 4 128 where
  tree := fun _ => NativePlan6.row
  leaves := fun _ => NativePlan6.prefix_correct

def plan_4_16 : PrefixPlan 4 16 where
  tree := fun _ => NativePlan2.row
  leaves := fun _ => NativePlan2.prefix_correct

def plan_4_256 : PrefixPlan 4 256 where
  tree := fun _ => NativePlan7.row
  leaves := fun _ => NativePlan7.prefix_correct

def plan_4_4 : PrefixPlan 4 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_4_64 : PrefixPlan 4 64 where
  tree := fun h => if h.val ∈ [0,1] then NativePlan4.row else NativePlan3.row
  leaves := by
    intro h j
    split_ifs
    · exact NativePlan4.prefix_correct j
    · exact NativePlan3.prefix_correct j

def plan_64_1 : PrefixPlan 64 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_64_128 : PrefixPlan 64 128 where
  tree := fun _ => NativePlan6.row
  leaves := fun _ => NativePlan6.prefix_correct

def plan_64_16 : PrefixPlan 64 16 where
  tree := fun _ => NativePlan2.row
  leaves := fun _ => NativePlan2.prefix_correct

def plan_64_256 : PrefixPlan 64 256 where
  tree := fun _ => NativePlan7.row
  leaves := fun _ => NativePlan7.prefix_correct

def plan_64_4 : PrefixPlan 64 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_64_64 : PrefixPlan 64 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

def plan_8_1 : PrefixPlan 8 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_8_128 : PrefixPlan 8 128 where
  tree := fun _ => NativePlan6.row
  leaves := fun _ => NativePlan6.prefix_correct

def plan_8_16 : PrefixPlan 8 16 where
  tree := fun _ => NativePlan2.row
  leaves := fun _ => NativePlan2.prefix_correct

def plan_8_256 : PrefixPlan 8 256 where
  tree := fun _ => NativePlan7.row
  leaves := fun _ => NativePlan7.prefix_correct

def plan_8_4 : PrefixPlan 8 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_8_64 : PrefixPlan 8 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

inductive Choice : (h n : Nat) → PrefixPlan h n → Prop where
  | bh1_bc1 : Choice 1 1 plan_1_1
  | bh1_bc4 : Choice 1 4 plan_1_4
  | bh1_bc16 : Choice 1 16 plan_1_16
  | bh1_bc64 : Choice 1 64 plan_1_64
  | bh1_bc128 : Choice 1 128 plan_1_128
  | bh1_bc256 : Choice 1 256 plan_1_256
  | bh2_bc1 : Choice 2 1 plan_2_1
  | bh2_bc4 : Choice 2 4 plan_2_4
  | bh2_bc16 : Choice 2 16 plan_2_16
  | bh2_bc64 : Choice 2 64 plan_2_64
  | bh2_bc128 : Choice 2 128 plan_2_128
  | bh2_bc256 : Choice 2 256 plan_2_256
  | bh4_bc1 : Choice 4 1 plan_4_1
  | bh4_bc4 : Choice 4 4 plan_4_4
  | bh4_bc16 : Choice 4 16 plan_4_16
  | bh4_bc64 : Choice 4 64 plan_4_64
  | bh4_bc128 : Choice 4 128 plan_4_128
  | bh4_bc256 : Choice 4 256 plan_4_256
  | bh8_bc1 : Choice 8 1 plan_8_1
  | bh8_bc4 : Choice 8 4 plan_8_4
  | bh8_bc16 : Choice 8 16 plan_8_16
  | bh8_bc64 : Choice 8 64 plan_8_64
  | bh8_bc128 : Choice 8 128 plan_8_128
  | bh8_bc256 : Choice 8 256 plan_8_256
  | bh16_bc1 : Choice 16 1 plan_16_1
  | bh16_bc4 : Choice 16 4 plan_16_4
  | bh16_bc16 : Choice 16 16 plan_16_16
  | bh16_bc64 : Choice 16 64 plan_16_64
  | bh16_bc128 : Choice 16 128 plan_16_128
  | bh16_bc256 : Choice 16 256 plan_16_256
  | bh32_bc1 : Choice 32 1 plan_32_1
  | bh32_bc4 : Choice 32 4 plan_32_4
  | bh32_bc16 : Choice 32 16 plan_32_16
  | bh32_bc64 : Choice 32 64 plan_32_64
  | bh32_bc128 : Choice 32 128 plan_32_128
  | bh32_bc256 : Choice 32 256 plan_32_256
  | bh64_bc1 : Choice 64 1 plan_64_1
  | bh64_bc4 : Choice 64 4 plan_64_4
  | bh64_bc16 : Choice 64 16 plan_64_16
  | bh64_bc64 : Choice 64 64 plan_64_64
  | bh64_bc128 : Choice 64 128 plan_64_128
  | bh64_bc256 : Choice 64 256 plan_64_256

structure Audited (h n : Nat) (plan : PrefixPlan h n) (bias softplus : Bool) : Prop where
  choice : Choice h n plan
  flags : bias = softplus

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.NativePlans
