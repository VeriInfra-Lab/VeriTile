import bench.tritonbench_g.matmul_triton2.Proofs.SourceParts
import bench.tritonbench_g.matmul_triton2.Proofs.FPReference
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
abbrev Word (α : Type) := VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α

def selectedOptions : DotOptions := {}
def symbolicPrimitives (α : Type) : Core.Primitives (Word α) where
  zero := .zero
  dot := fun left right carry => .dot (some .fp32) selectedOptions left right carry

variable {α : Type} [Inhabited α]

structure RuntimeInvariant (p : Core.Parameters) (initial : State (Word α))
    (block : Nat) (current : State (Word α)) : Prop where
  memory : current.mem = initial.mem
  pids : current.pids = initial.pids
  metadata : ∀ name n, (name,n) ∈ VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters p →
    current.regs .nat [] name = some (fun _ => n)
  offm : current.regs .nat [p.blockRows] "offs_am" = some (fun i => Core.row p initial i.1)
  offn : current.regs .nat [p.blockColumns] "offs_bn" = some (fun i => Core.column p initial i.1)
  offk : current.regs .nat [p.blockInner] "offs_k" = some (fun i => i.1.val)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input,Core.aOffset p initial block i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" =
    some (fun i => (p.weight,Core.bOffset p initial block i))
  acc : current.regs .real [p.blockRows,p.blockColumns] "accumulator" =
    some (Core.accumulation (symbolicPrimitives α) p initial block)

theorem preludeInvariant (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (s : State (Word α))
    (small : ∀ name n, (name,n) ∈ VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters p → n < 2^64)
    (pidSmall : s.pids 0 < 2^64)
    (bmSmall : p.blockRows ≤ 2^64) (bnSmall : p.blockColumns ≤ 2^64)
    (bkSmall : p.blockInner ≤ 2^64) (groupSmall : p.groupRows < 2^64) :
    RuntimeInvariant p s 0 (preludeExecution rest P p s).val := by
  have mod0 : p.rows % 2^64 = p.rows :=
    Nat.mod_eq_of_lt (small "M" p.rows (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod1 : p.innerSize % 2^64 = p.innerSize :=
    Nat.mod_eq_of_lt (small "K" p.innerSize (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod2 : p.columns % 2^64 = p.columns :=
    Nat.mod_eq_of_lt (small "N" p.columns (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod3 : p.inputRow % 2^64 = p.inputRow :=
    Nat.mod_eq_of_lt (small "stride_am" p.inputRow (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod4 : p.inputInner % 2^64 = p.inputInner :=
    Nat.mod_eq_of_lt (small "stride_ak" p.inputInner (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod5 : p.weightInner % 2^64 = p.weightInner :=
    Nat.mod_eq_of_lt (small "stride_bk" p.weightInner (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod6 : p.weightColumn % 2^64 = p.weightColumn :=
    Nat.mod_eq_of_lt (small "stride_bn" p.weightColumn (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod7 : p.outputRow % 2^64 = p.outputRow :=
    Nat.mod_eq_of_lt (small "stride_cm" p.outputRow (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  have mod8 : p.outputColumn % 2^64 = p.outputColumn :=
    Nat.mod_eq_of_lt (small "stride_cn" p.outputColumn (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters]))
  norm_num only [Nat.reducePow] at mod0 mod1 mod2 mod3 mod4 mod5 mod6 mod7 mod8
  have modPid : s.pids 0 % 18446744073709551616 = s.pids 0 := Nat.mod_eq_of_lt pidSmall
  have laneM (i : Fin p.blockRows) : i.val % 18446744073709551616 = i.val :=
    Nat.mod_eq_of_lt (lt_of_lt_of_le i.isLt bmSmall)
  have laneN (i : Fin p.blockColumns) : i.val % 18446744073709551616 = i.val :=
    Nat.mod_eq_of_lt (lt_of_lt_of_le i.isLt bnSmall)
  have laneK (i : Fin p.blockInner) : i.val % 18446744073709551616 = i.val :=
    Nat.mod_eq_of_lt (lt_of_lt_of_le i.isLt bkSmall)
  have groupMod : (Core.groupSize p s) % 18446744073709551616 = Core.groupSize p s :=
    Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.min_le_right _ _) groupSmall)
  dsimp only [Core.groupSize,Core.blocksM,Core.firstRow,Core.group,Core.width,Core.blocksN] at groupMod
  have min_lt_form (a b : Nat) : (if a<b then a else b)=min a b := by
    split_ifs with h
    · exact (Nat.min_eq_left (Nat.le_of_lt h)).symm
    · exact (Nat.min_eq_right (Nat.le_of_not_lt h)).symm
  constructor
  · exact prelude_memory rest P p s
  · exact prelude_pids rest P p s
  · intro name n member
    simpa only [Nat.mod_eq_of_lt (small name n member)] using prelude_metadata rest P p s name n member
  all_goals simp [preludeExecution,Core.accumulation,symbolicPrimitives,
    Core.aOffset,Core.bOffset,Core.cOffset,Core.row,Core.column,
    Core.rowBlock,Core.columnBlock,Core.blocksN,Core.blocksM,Core.width,Core.firstRow,Core.group,Core.groupSize,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Int.toNat_emod,min_lt_form,groupMod,modPid,laneM,laneN,laneK,mod0,mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8,Nat.add_comm]
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,min_lt_form,groupMod,modPid,laneM,laneN,laneK,mod0,mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8,Nat.add_comm]

  · exact Or.inl (Nat.add_comm _ _)
  · exact Or.inl (Or.inl (Nat.add_comm _ _))

#axiomsClean preludeInvariant
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
