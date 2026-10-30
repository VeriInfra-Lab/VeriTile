import bench.tritonbench_g.int8_dequant_matmul.IO
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSourceResult
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IODataBinding

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOInvocation
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

@[simp] theorem pid0 (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c) (base : State (Sum F H)) :
    ((IO.invocation q s c hasBias divisor p).initial base).pids 0=p.1.val := by
  simp [IO.invocation,IntegerDotInvocation.initial,Invocation.initial,bindRegisters]

@[simp] theorem pid1 (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c) (base : State (Sum F H)) :
    ((IO.invocation q s c hasBias divisor p).initial base).pids 1=p.2.val := by
  simp [IO.invocation,IntegerDotInvocation.initial,Invocation.initial,bindRegisters]

@[simp] theorem divisor (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c) (base : State (Sum F H)) :
    ((IO.invocation q s c hasBias divisor p).initial base).regs .fp32 [] "divfactor"=
      some (fun _=>Sum.inl divisor) := by
  simp [IO.invocation,IntegerDotInvocation.initial,Invocation.initial,bindRegisters]

theorem run_eq (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c)
    (base : State (Sum F H)) :
    (IO.invocation q s c hasBias divisor p).run base=
      Quote.execute (Quote.floating q) q s.M s.N (Quote.source s c hasBias)
        ((IO.invocation q s c hasBias divisor p).initial base) :=
  Quote.SharedInteger.exec_eq _ _ _ _ _ _

theorem data (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals base) :
    IOInputReads.DataCells s c hasBias vals ((IO.invocation q s c hasBias divisor p).initial base) := by
  intro i j notC active
  simpa only [IntegerDotInvocation.initial_mem] using cells i j notC active

theorem success (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c)
    (base : State (Sum F H)) :
    ∃ final,(IO.invocation q s c hasBias divisor p).run base=some final := by
  obtain ⟨final,ran,_⟩:=Quote.fullResult q s c hasBias
    ((IO.invocation q s c hasBias divisor p).initial base) divisor (by simp)
  exact ⟨final,(run_eq q s c hasBias divisor p base).trans ran⟩

theorem frame (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base final : State (Sum F H))
    (ran : (IO.invocation q s c hasBias divisor p).run base=some final) :
    (IOLayout.memory F H s c hasBias).Frame vals base final := by
  rw [run_eq] at ran
  have result:=IOSourceResult.frame q s c hasBias vals
    ((IO.invocation q s c hasBias divisor p).initial base) (by simpa using p.1.isLt)
    divisor (by simp) final ran
  simpa only [MemoryIO.Frame,IntegerDotInvocation.initial_mem] using result

theorem cell (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base final : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals base)
    (selected : Selection s c) (plans : q.plans=Quote.selectedPlan s c selected)
    (ran : (IO.invocation q s c hasBias divisor p).run base=some final) (x : MatrixCell s) :
    final.mem "C" (x.1.val*s.N+x.2.val)=
      if p.1.val=(IOPrograms.owner s c x).val then
        Cell.mk .fp16 (.inr (if c.S=1 then
          Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x p.2
          else q.ops.hadd (Quote.asH q.ops ((base.mem "C" (x.1.val*s.N+x.2.val)).read .fp16))
            (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x p.2)))
      else base.mem "C" (x.1.val*s.N+x.2.val) := by
  rw [run_eq] at ran
  have result:=IOSourceResult.cell q s c hasBias vals
    ((IO.invocation q s c hasBias divisor p).initial base) (data q s c hasBias divisor p vals base cells)
    selected plans (by simpa using p.1.isLt) (by simpa using p.2.isLt) divisor (by simp) final ran x
  simpa only [pid0,pid1,IntegerDotInvocation.initial_mem] using result

#axiomsClean run_eq
#axiomsClean success
#axiomsClean frame
#axiomsClean cell
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOInvocation
