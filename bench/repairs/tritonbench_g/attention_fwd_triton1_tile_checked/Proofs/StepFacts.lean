import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Step

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedStep
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody
noncomputable section
set_option maxHeartbeats 8000000

theorem updated_memory [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (updated c scale head chunk storeFlag ifcond carry s).mem=
      (saved c head chunk storeFlag carry s).mem := by
  cases ifcond <;> by_cases first : chunk=0 <;> simp [updated,prepared,first]

theorem finished_input [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α))
    (r : RegionName) (off : Nat) (ho : r≠"o") (hh : r≠"h") :
    (finished c scale head chunk storeFlag ifcond carry s).mem r off=s.mem r off := by
  rw [finished,scatter_other _ _ _ _ _ _ r off ho,updated_memory]
  exact saved_input c head chunk storeFlag carry s r off hh

theorem finished_carry [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (finished c scale head chunk storeFlag ifcond carry s).regs .real [c.D,c.D] "b_h"=
      some (nextCarry c (read s "k") (read s "v") head chunk carry ifcond) := by
  simp only [finished,scatter_regs,updated_carry]

theorem finished_head [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (finished c scale head chunk storeFlag ifcond carry s).regs .int [] "i_bh"=
      s.regs .int [] "i_bh" := by
  cases ifcond <;> by_cases first : chunk=0 <;>
    simp [finished,updated,prepared,saved_regs,pointers,first]

theorem finished_pids [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D) (s : State (Term α)) :
    (finished c scale head chunk storeFlag ifcond carry s).pids=s.pids := by
  cases ifcond <;> cases storeFlag <;> by_cases first : chunk=0 <;>
    simp [finished,updated,prepared,saved,pointers,first]

#axiomsClean finished_input
#axiomsClean finished_carry
#axiomsClean finished_head
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedStep
