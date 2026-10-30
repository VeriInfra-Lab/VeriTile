import bench.tritonbench_g.matmul_leakyrelu_fp8.Execution
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Schedule

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 3000000

@[simp] theorem qDiv_ofNat (n m : Nat) : qDiv (n : Int) (m : Int) = ((n/m : Nat) : Int) := by
  simp [qDiv]

@[simp] theorem qMod_ofNat (n m : Nat) : qMod (n : Int) (m : Int) = ((n%m : Nat) : Int) := by
  rw [qMod,qDiv_ofNat]
  simpa only [Int.natCast_emod,Int.natCast_ediv] using (Int.emod_def (n : Int) (m : Int)).symm

theorem rows_eq (p : Problem) : rows p = p.rows := by
  simp only [rows,Problem.rows,RealMath.blocks]
  split_ifs <;> omega

theorem columns_eq (p : Problem) : columns p = p.columns := by
  simp only [columns,Problem.columns,RealMath.blocks]
  split_ifs <;> omega

theorem trips_eq (p : Problem) : trips p = p.kTiles := by
  simp only [trips,Problem.kTiles,Storage.blockK,RealMath.blocks]
  split_ifs <;> omega

theorem group_eq (p : Problem) : group p = p.group := by rw [group,rows_eq]; rfl

theorem firstM_eq {α : Type} (p : Problem) (s : State α) :
    firstM p s = (p.groupBase (s.pids 0) : Int) := by
  simp only [firstM,perGroup,group_eq,columns_eq,←Int.natCast_mul,qDiv_ofNat]
  simp [Problem.groupBase,Schedule.groupBase]

theorem group_base_bound {α : Type} (p : Problem) (s : State α)
    (ht : s.pids 0 < p.tiles) : p.groupBase (s.pids 0) < p.rows := by
  have hb := Nat.div_mul_le_self (s.pids 0) (p.group*p.columns)
  change s.pids 0/(p.group*p.columns)*p.group < p.rows
  change s.pids 0 < p.rows*p.columns at ht
  nlinarith

theorem groupSize_eq {α : Type} (p : Problem) (s : State α)
    (ht : s.pids 0 < p.tiles) : groupSize p s = (p.groupSize (s.pids 0) : Int) := by
  rw [groupSize,rows_eq,firstM_eq,group_eq]
  change min ((p.rows : Int)-(p.groupBase (s.pids 0) : Int)) (p.group : Int) =
    ((min (p.rows-p.groupBase (s.pids 0)) p.group : Nat) : Int)
  rw [←Int.ofNat_sub (Nat.le_of_lt (group_base_bound p s ht))]
  omega

theorem pidM_eq {α : Type} (p : Problem) (s : State α)
    (ht : s.pids 0 < p.tiles) : pidM p s = (p.tileRow (s.pids 0) : Int) := by
  simp only [pidM,firstM_eq,groupSize_eq p s ht,perGroup,group_eq,columns_eq,←Int.natCast_mul,qMod_ofNat]
  simp [Problem.tileRow]

theorem pidN_eq {α : Type} (p : Problem) (s : State α)
    (ht : s.pids 0 < p.tiles) : pidN p s = (p.tileColumn (s.pids 0) : Int) := by
  simp only [pidN,groupSize_eq p s ht,perGroup,group_eq,columns_eq,←Int.natCast_mul,qMod_ofNat,qDiv_ofNat]
  rfl

theorem row_eq {α : Type} (p : Problem) (s : State α) (i : Fin 128)
    (ht : s.pids 0 < p.tiles) : row p s i = (p.row (s.pids 0) i.val : Int) := by
  simp [row,pidM_eq p s ht,Problem.row]

theorem col_eq {α : Type} (p : Problem) (s : State α) (j : Fin 256)
    (ht : s.pids 0 < p.tiles) : col p s j = (p.column (s.pids 0) j.val : Int) := by
  simp [col,pidN_eq p s ht,Problem.column]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
