import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.TypedProgram
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Geometry
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.InstructionBackend

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def Inputs {α : Type} (b : Buffers) (s : State α) (memory : Memory α) : Prop :=
  (∀ o, s.mem "A" o = Cell.mk .real (memory (b.A.storage,b.A.offset+o))) ∧
  (∀ o, s.mem "B" o = Cell.mk .real (memory (b.B.storage,b.B.offset+o)))

theorem safeK_eq (p : Problem) (k lane : Nat) : safeK p k lane = (clamp p.K (k*64+lane) : Int) := by
  have he : (k : Int)*64+(lane : Int) = ((k*64+lane : Nat) : Int) := by simp
  simp only [safeK,he,clamp]
  split_ifs <;> omega

theorem aOffset_eq {α : Type} (p : Problem) (b : Buffers) (s : State α)
    (k : Nat) (i : AIndex) (ht : s.pids 0 < p.tiles) :
    aOffset p b s k i =
      (p.row (s.pids 0) i.1.val%p.M)*b.A.rowStride + clamp p.K (k*64+i.2.1.val)*b.A.columnStride := by
  simp only [aOffset,row_eq p s i.1 ht,qMod_ofNat,safeK_eq,←Int.natCast_mul,←Int.natCast_add]
  rfl

theorem bOffset_eq {α : Type} (p : Problem) (b : Buffers) (s : State α)
    (k : Nat) (i : BIndex) (ht : s.pids 0 < p.tiles) :
    bOffset p b s k i =
      clamp p.K (k*64+i.1.val)*b.B.rowStride + (p.column (s.pids 0) i.2.1.val%p.N)*b.B.columnStride := by
  simp only [bOffset,col_eq p s i.2.1 ht,qMod_ofNat,safeK_eq,←Int.natCast_mul,←Int.natCast_add]
  rfl

theorem active_eq {α : Type} (p : Problem) (s : State α) (i : CIndex)
    (ht : s.pids 0 < p.tiles) :
    active p s i ↔ Source.active p (s.pids 0) (i.1,i.2.1) := by
  simp [active,row_eq p s i.1 ht,col_eq p s i.2.1 ht,Source.active]

theorem cOffset_eq {α : Type} (p : Problem) (b : Buffers) (s : State α) (i : CIndex)
    (ht : s.pids 0 < p.tiles) (hi : active p s i) :
    cOffset p b s i = p.row (s.pids 0) i.1.val*b.C.rowStride +
      p.column (s.pids 0) i.2.1.val*b.C.columnStride := by
  unfold cOffset
  rw [if_pos hi.1,if_pos hi.2]
  simp only [row_eq p s i.1 ht,col_eq p s i.2.1 ht,
    ←Int.natCast_mul,←Int.natCast_add]
  change b.C.rowStride*p.row (s.pids 0) i.1.val +
      b.C.columnStride*p.column (s.pids 0) i.2.1.val = _
  simp only [Nat.mul_comm]

theorem loadA_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (s : State α)
    (memory : Memory α) (inputs : Inputs b s memory) (k : Nat) (i : AIndex)
    (ht : s.pids 0 < p.tiles) :
    aValue M p b s k i = Source.loadA (sourceArithmetic ops) p b memory (s.pids 0) k i.1 i.2.1 := by
  have mask : ((k : Int)*64+(i.2.1.val : Int) < (p.K : Int)) ↔ k*64+i.2.1.val < p.K := by omega
  simp only [aValue,mask,Source.loadA,Problem.k,Storage.blockK]
  split_ifs
  · rw [inputs.1,Cell.read_same,law.loadValue,aOffset_eq p b s k i ht]
    simp [Tensor.address,Nat.add_assoc]
  · exact law.loadZero

theorem loadB_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (s : State α)
    (memory : Memory α) (inputs : Inputs b s memory) (k : Nat) (i : BIndex)
    (ht : s.pids 0 < p.tiles) :
    bValue M p b s k i = Source.loadB (sourceArithmetic ops) p b memory (s.pids 0) k i.1 i.2.1 := by
  have mask : ((k : Int)*64+(i.1.val : Int) < (p.K : Int)) ↔ k*64+i.1.val < p.K := by omega
  simp only [bValue,mask,Source.loadB,Problem.k,Storage.blockK]
  split_ifs
  · rw [inputs.2,Cell.read_same,law.loadValue,bOffset_eq p b s k i ht]
    simp [Tensor.address,Nat.add_assoc]
  · exact law.loadZero

def sourceAccumulator (ops : Operations α) (p : Problem) (b : Buffers)
    (memory : Memory α) (tile count : Nat) : CIndex → α :=
  fun i => Source.accumulator (sourceArithmetic ops) p b memory tile count i.1 i.2.1

theorem update_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (s : State α)
    (memory : Memory α) (inputs : Inputs b s memory) (k : Nat) (acc : CIndex → α)
    (ht : s.pids 0 < p.tiles) (i : CIndex) :
    update M (selectedDot ops) p b s k acc i =
      block ops (Source.loadA (sourceArithmetic ops) p b memory (s.pids 0) k i.1)
        (fun lane => Source.loadB (sourceArithmetic ops) p b memory (s.pids 0) k lane i.2.1) (acc i) := by
  rw [update,selected_dot]
  congr 1
  · funext lane
    exact loadA_source M ops law p b s memory inputs k (i.1,lane,PUnit.unit) ht
  · funext lane
    exact loadB_source M ops law p b s memory inputs k (lane,i.2.1,PUnit.unit) ht

theorem update_source_accumulator {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (s : State α)
    (memory : Memory α) (inputs : Inputs b s memory) (k : Nat) (ht : s.pids 0 < p.tiles) :
    update M (selectedDot ops) p b s k (sourceAccumulator ops p b memory (s.pids 0) k) =
      sourceAccumulator ops p b memory (s.pids 0) (k+1) := by
  funext i
  rw [update_source M ops law p b s memory inputs k _ ht i]
  rfl

theorem fold_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (s : State α)
    (memory : Memory α) (inputs : Inputs b s memory) (ht : s.pids 0 < p.tiles)
    (cur stop : Nat) (hc : cur ≤ stop) :
    foldRange (update M (selectedDot ops) p b s) cur stop 1
      (sourceAccumulator ops p b memory (s.pids 0) cur) =
      sourceAccumulator ops p b memory (s.pids 0) stop := by
  by_cases hh : cur < stop
  · rw [foldRange]
    simp only [show (1 : Nat) ≠ 0 by decide,if_false,hh,if_true]
    rw [update_source_accumulator M ops law p b s memory inputs cur ht]
    exact fold_source M ops law p b s memory inputs ht (cur+1) stop (by omega)
  · have same : cur = stop := by omega
    subst cur
    rw [foldRange]
    simp
termination_by stop-cur

theorem accumulation_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (s : State α)
    (memory : Memory α) (inputs : Inputs b s memory) (ht : s.pids 0 < p.tiles) :
    accumulation M (selectedDot ops) p b s = sourceAccumulator ops p b memory (s.pids 0) p.kTiles := by
  unfold accumulation
  rw [law.initialZero,trips_eq]
  exact fold_source M ops law p b s memory inputs ht 0 p.kTiles (Nat.zero_le _)

theorem output_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (s : State α)
    (memory : Memory α) (inputs : Inputs b s memory) (ht : s.pids 0 < p.tiles) (i : CIndex) :
    outputValue M (selectedDot ops) p b s i =
      ops.castHalf (Checked.activate ops p.enabled (sourceAccumulator ops p b memory (s.pids 0) p.kTiles i)) := by
  rw [outputValue,law.halfCast,accumulation_source M ops law p b s memory inputs ht]
  simp [activation,selectedDot,law.slope,Checked.activate]

#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
