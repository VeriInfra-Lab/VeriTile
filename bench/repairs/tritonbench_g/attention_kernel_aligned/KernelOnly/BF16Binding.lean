import bench.repairs.tritonbench_g.attention_kernel_aligned.KernelOnly.BF16Model
import bench.repairs.tritonbench_g.attention_kernel_aligned.KernelOnly.BF16Specification
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.KernelOnly

open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
open VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly
open VeriTile.Bench.Repairs.AttentionKernelAligned.BF16 Source Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]
def operandReader (s : State α) (r : RegionName) (off : Nat) : α := (s.mem r off).read .bf16
def toSource (p : Parameters) (a : Specification.Accumulator α p) : Source.Accumulator α p :=
  ⟨a.maximum,a.denominator,a.numerator⟩
theorem advance_check (P : Primitives α) (p : Parameters) (s : State α) (k : Nat)
    (a : Specification.Accumulator α p) :
    Source.advance P p s k (toSource p a)=toSource p
      (Specification.advance P p (operandReader s) (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) k a) := by rfl
theorem rounds_check (P : Primitives α) (p : Parameters) (s : State α) (k : Nat) :
    Source.rounds P p s k=toSource p
      (Specification.rounds P p (operandReader s) (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) k) := by
  induction k with
  | zero=>rfl
  | succ k ih=>rw [Source.rounds,Specification.rounds,ih,advance_check]
theorem source_algorithm (P : Primitives α) (p : Parameters) (s : State α) :
    Whole.answer P p s=Specification.outputValue P p (operandReader s)
      (s.pids 0) (s.pids 1/p.heads) (s.pids 1%p.heads) := by
  funext i
  unfold Whole.answer Source.outputValue Specification.outputValue
  rw [rounds_check]
  rfl
theorem algorithm_at (P : Primitives α) (c : Call) (read : RegionName → Nat → α)
    (x b h : Nat) (hh : h<(parameters c).heads) :
    Whole.answer P (parameters c)
      (start (grid (parameters c)) (x,b*(parameters c).heads+h,0) (entry read))=
    Specification.outputValue P (parameters c) read x b h := by
  rw [source_algorithm]
  have hp : 0<(parameters c).heads := by omega
  have hd : (b*(parameters c).heads+h)/(parameters c).heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hm : (b*(parameters c).heads+h)%(parameters c).heads=h := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have reader : operandReader
      (start (grid (parameters c)) (x,b*(parameters c).heads+h,0) (entry read))=read := by
    funext r off
    rfl
  rw [reader]
  simp [start,ids,hd,hm]

#axiomsClean advance_check
#axiomsClean rounds_check
#axiomsClean source_algorithm
#axiomsClean algorithm_at
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.KernelOnly
