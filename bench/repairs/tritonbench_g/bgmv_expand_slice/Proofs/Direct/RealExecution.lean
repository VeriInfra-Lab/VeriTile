import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.RealConfig
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Geometry
import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealProgram

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.RealExecution
open VeriTile.Triton RealConfig
open scoped BigOperators
namespace Old
abbrev source_executes := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealProgram.source_executes
abbrev sentinel_skips := VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealProgram.sentinel_skips
end Old
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Independent K-term sum of entry storage, before any source statements.
The separate host proof substitutes original caller logical coordinates. -/
def formula (h : Call) (p : Geometry.Program h) (u : BlockState) (slot : Nat) : ℝ :=
  let dot := ∑ k : Fin h.K,
    u.readMem "X" (p.batch*h.K+k.val) *
    u.readMem "W" (slot*(h.N*h.K)+p.row*h.K+k.val)
  if h.add then dot+u.readMem "Y" (Geometry.outputOffset h p.batch p.row 0) else dot

def writes (h : Call) (p : Geometry.Program h) (slot : Int) (r : RegionName) (off : Nat) : Prop :=
  slot≠-1 ∧ r="Y" ∧ off=Geometry.outputOffset h p.batch p.row 0

theorem source_executes (h : Call) (p : Geometry.Program h) (u : BlockState)
    (batch : u.pids 1=p.batch) (row : u.pids 0=p.row)
    (slot : Int) (valid : slot=-1 ∨ 0≤slot)
    (metadata : u.readMemValue .int "Indices" p.batch=slot) :
    ∃ v, exec (Kernels.source h).eraseDType.toAlgKernel u=some v ∧
      (slot=-1 → v.mem=u.mem) ∧
      (slot≠-1 → v.readMem "Y" (Geometry.outputOffset h p.batch p.row 0)=formula h p u slot.toNat) ∧
      (∀ r off, ¬writes h p slot r off → v.mem r off=u.mem r off) := by
  have positive : 0<h.N := by have := p.rowInside; omega
  have length : (parameters h).length=1 := Kernels.split_length h positive
  have metadata' : u.readMemValue .int (Region.cast (parameters h).indices) (u.pids 1)=slot := by
    simpa only [parameters,Region.cast,batch] using metadata
  rcases valid with sentinel | active
  · rw [sentinel] at metadata'
    obtain ⟨v,he,frame⟩ := Old.sentinel_skips (parameters h) h.inputType h.weightType h.outputType
      (Kernels.hostCast h.inputType) Bool.false h.even h.add u metadata'
    refine ⟨v,?_,fun _ => frame,?_,?_⟩
    · simpa only [RealConfig.source_parameters] using he
    · intro impossible; exact False.elim (impossible sentinel)
    · intro r off _; exact congrFun (congrFun frame r) off
  · lift slot to Nat using active
    have nonnegative : (slot:Int)≠-1 := by omega
    obtain ⟨v,he,values,frame⟩ := Old.source_executes (parameters h) h.inputType h.weightType h.outputType
      (Kernels.hostCast h.inputType) Bool.false h.even h.add u slot metadata'
      (by change 0<256; decide) (by change 0<1; decide) h.rankCovered positive (by change ("W":RegionName)≠"Y"; decide)
    refine ⟨v,?_,fun impossible => False.elim (nonnegative impossible),?_,?_⟩
    · simpa only [RealConfig.source_parameters] using he
    · intro _
      have value := values 0 (by rw [length]; decide)
      simp only [VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealProgram.formula,
        VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSource.Parameters.geometry,
        VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoop.Geometry.outputAddress,
        length] at value
      simpa only [parameters,batch,row,formula,Geometry.outputOffset,
        Nat.mul_one,Nat.zero_mul,Nat.add_zero,Int.toNat_natCast,Nat.mul_comm slot (h.N*h.K)] using value
    · intro r off untouched
      apply frame r off
      intro g hg eq
      have zero : g=0 := by
        change g<(parameters h).length at hg
        rw [length] at hg; omega
      subst g
      apply untouched
      refine ⟨nonnegative,?_,?_⟩
      · exact eq.1.symm
      · have same := eq.2.symm
        simp only [VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetupSource.Parameters.geometry,
          VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoop.Geometry.outputAddress,
          length] at same
        simpa only [parameters,batch,row,Geometry.outputOffset,Nat.mul_one,Nat.zero_mul,Nat.add_zero] using same

#axiomsClean source_executes
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.RealExecution
