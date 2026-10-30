import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealBackwardComplete
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealForwardGrid
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.BackwardGridAddresses
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 6000000

def rowAt (c : Shape) (μ : RegionName → Nat → ℝ) (r : RegionName) (h i : Nat) : ℝ :=
  μ r (h*c.rows+i)
def probabilityAt (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (h i j : Nat) : ℝ :=
  backwardProbability c.dim scale (tensorAt c μ "Q" h) (tensorAt c μ "K" h) (rowAt c μ "M" h) i j
def dsAt (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (h i j : Nat) : ℝ :=
  backwardScoreGradient c.dim scale (tensorAt c μ "Q" h) (tensorAt c μ "K" h)
    (tensorAt c μ "V" h) (tensorAt c μ "NewDO" h) (rowAt c μ "M" h) (rowAt c μ "Delta" h) i j

def tiledQ (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (i : Coord c) : ℝ :=
  ∑ b ∈ Finset.range c.blocks,if b ≤ i.2.1.val / 128 then
    ∑ j ∈ Finset.range 128,dsAt c scale μ i.1.val i.2.1.val (b*128+j)*tensorAt c μ "K" i.1.val (b*128+j) i.2.2.val
    else 0
def tiledK (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (i : Coord c) : ℝ :=
  ∑ t ∈ Finset.range (c.blocks-i.2.1.val/128),∑ j ∈ Finset.range 128,
    dsAt c scale μ i.1.val ((i.2.1.val/128+t)*128+j) i.2.1.val*
      tensorAt c μ "Q" i.1.val ((i.2.1.val/128+t)*128+j) i.2.2.val
def tiledV (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (i : Coord c) : ℝ :=
  ∑ t ∈ Finset.range (c.blocks-i.2.1.val/128),∑ j ∈ Finset.range 128,
    probabilityAt c scale μ i.1.val ((i.2.1.val/128+t)*128+j) i.2.1.val*
      tensorAt c μ "NewDO" i.1.val ((i.2.1.val/128+t)*128+j) i.2.2.val

theorem bAt_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (r : RegionName) :
    bAt (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid mem) μ r=tensorAt c μ r pid.1 := by
  funext row d
  simp only [bAt,Grid.backwardHead_eq]
  rfl

theorem bRow_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (r : RegionName) :
    bRow (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid mem) μ r=rowAt c μ r pid.1 := by
  funext row
  simp [bRow,rowAt,backwardParameters,Grid.entry,Grid.ids,Shape.rows,Nat.mul_comm]

theorem bP_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) :
    bP (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid mem) μ=probabilityAt c scale μ pid.1 := by
  funext i j
  simp only [bP,bAt_entry,bRow_entry]
  rfl

theorem bDS_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) :
    bDS (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid mem) μ=dsAt c scale μ pid.1 := by
  funext i j
  simp only [bDS,bAt_entry,bRow_entry]
  rfl

theorem queryContribution_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (k q : Nat) (i : BLane (backwardParameters c scale)) :
    queryContribution (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid mem) μ k q i=
      ∑ j ∈ Finset.range 128,dsAt c scale μ pid.1 (q*128+i.1.val) (k*128+j)*tensorAt c μ "K" pid.1 (k*128+j) i.2.1.val := by
  simp only [queryContribution,bDS_entry,bAt_entry]
  rfl

theorem keyContribution_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (k q : Nat) (i : BLane (backwardParameters c scale)) :
    keyContribution (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid mem) μ k q i=
      ∑ j ∈ Finset.range 128,dsAt c scale μ pid.1 (q*128+j) (k*128+i.1.val)*tensorAt c μ "Q" pid.1 (q*128+j) i.2.1.val := by
  simp only [keyContribution,bDS_entry,bAt_entry]
  rfl

theorem valueContribution_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (k q : Nat) (i : BLane (backwardParameters c scale)) :
    valueContribution (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid mem) μ k q i=
      ∑ j ∈ Finset.range 128,probabilityAt c scale μ pid.1 (q*128+j) (k*128+i.1.val)*tensorAt c μ "NewDO" pid.1 (q*128+j) i.2.1.val := by
  simp only [valueContribution,bP_entry,bAt_entry]
  rfl

#axiomsClean queryContribution_entry
#axiomsClean keyContribution_entry
#axiomsClean valueContribution_entry
#specNonCircular tiledQ avoiding [FP.Structural.exec,backwardRounds,backwardFinalMemory,queryContribution]
#specNonCircular tiledK avoiding [FP.Structural.exec,backwardRounds,backwardFinalMemory,keyContribution]
#specNonCircular tiledV avoiding [FP.Structural.exec,backwardRounds,backwardFinalMemory,valueContribution]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
