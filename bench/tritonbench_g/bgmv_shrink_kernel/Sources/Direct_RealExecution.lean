import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_Geometry
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_RealProgram

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.RealExecution
open VeriTile.Triton
open scoped BigOperators
namespace Old
abbrev Parameters := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSource.Parameters
abbrev source_executes := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealProgram.source_executes
abbrev sentinel_skips := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealProgram.sentinel_skips
end Old
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def parameters (h : Call) : Old.Parameters where
  input := "X"
  weights := "W"
  output := "Y"
  indices := ⟨"Indices"⟩
  N := h.N
  K := h.K
  scaling := h.scaling
  inputBatchStride := h.K
  inputStride := 1
  slotStride := h.N*h.K
  rowStride := h.K
  colStride := 1
  outputBatchStride := h.columns
  outputStride := 1
  BN := h.BN
  BK := 256
  splits := 64

theorem source_parameters (h : Call) :
    (parameters h).source h.storage h.outputType Bool.false=Kernels.source h := rfl

/-- This split's masked summands read immutable entry X/W. The independent
whole-call specification below instead sums directly over Fin K. -/
def contribution (h : Call) (s : BlockState) (b slot n split : Nat) : ℝ :=
  h.scaling * ∑ c ∈ Finset.range h.cycles,∑ e : Fin 256,
    let k := Geometry.rank split c e.val
    if k<h.K then s.readMem "X" (Geometry.inputOffset h b k)*
      s.readMem "W" (Geometry.weightOffset h slot n k) else 0

def writes (h : Call) (p : Geometry.Program h) (slot : Int) (r : RegionName) (off : Nat) : Prop :=
  slot≠-1 ∧ r="Y" ∧ ∃ n : Fin h.N,off=Geometry.outputOffset h p.batch.val n.val

/-- Complete original signed program, all loop iterations and atomic lanes.
Nothing about execution success or computed values is an input hypothesis. -/
theorem source_executes (h : Call) (p : Geometry.Program h) (u : BlockState)
    (batch : u.pids 1=p.batch.val) (split : u.pids 0=p.split.val)
    (slot : Int) (valid : slot=-1 ∨ 0≤slot)
    (metadata : u.readMemValue .int "Indices" p.batch.val=slot) :
    ∃ v,exec (Kernels.source h).eraseDType.toAlgKernel u=some v ∧
      (slot=-1 → v.mem=u.mem) ∧
      (slot≠-1 → ∀ n : Fin h.N,v.readMem "Y" (Geometry.outputOffset h p.batch.val n.val)=
        u.readMem "Y" (Geometry.outputOffset h p.batch.val n.val)+
          contribution h u p.batch.val slot.toNat n.val p.split.val) ∧
      (∀ r off,¬writes h p slot r off → v.mem r off=u.mem r off) := by
  have metadata' : u.readMemValue .int (Region.cast (parameters h).indices) (u.pids 1)=slot := by
    simpa only [parameters,Region.cast,batch] using metadata
  rcases valid with sentinel | active
  · rw [sentinel] at metadata'
    obtain ⟨v,he,frame⟩ := Old.sentinel_skips (parameters h) h.storage h.outputType Bool.false u metadata'
    exact ⟨v,by simpa only [source_parameters] using he,fun _ => frame,
      fun impossible => False.elim (impossible sentinel),fun r off _ => congrFun (congrFun frame r) off⟩
  · lift slot to Nat using active
    have nonnegative : (slot:Int)≠-1 := by omega
    obtain ⟨v,he,values,frame⟩ := Old.source_executes (parameters h) h.storage h.outputType Bool.false u slot metadata'
      (by change 0<256*64; decide) (by change 0<1; decide) h.rowsCovered
    refine ⟨v,by simpa only [source_parameters] using he,
      fun impossible => False.elim (nonnegative impossible),?_,?_⟩
    · intro _ n
      have value := values n.val n.isLt
      have counts : (h.K+16384-1)/16384=h.cycles := by
        unfold Call.cycles
        congr 1
      change v.readMem "Y" (u.pids 1*h.columns+n.val*1)=
        u.readMem "Y" (u.pids 1*h.columns+n.val*1)+
        h.scaling*∑ c ∈ Finset.range ((h.K+16384-1)/16384),∑ e : Fin 256,
          if c*16384+(e.val+u.pids 0*256)<h.K then
            u.readMem "X" (u.pids 1*h.K+(c*16384+(e.val+u.pids 0*256)))*
            u.readMem "W" ((h.N*h.K)*slot+n.val*h.K+(c*16384+(e.val+u.pids 0*256))*1)
          else 0 at value
      simpa only [counts,contribution,Geometry.outputOffset,Geometry.inputOffset,Geometry.weightOffset,
        Geometry.rank,batch,split,Nat.mul_one,Int.toNat_natCast,
        Nat.mul_comm (h.N*h.K) slot] using value
    · intro r off outside
      apply frame r off
      intro n hn equal
      apply outside
      refine ⟨nonnegative,equal.1.symm,⟨n.val,hn⟩,?_⟩
      simpa only [VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealTail.offset,
        parameters,batch,Geometry.outputOffset,Nat.mul_one] using equal.2.symm

#axiomsClean source_executes
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.RealExecution
