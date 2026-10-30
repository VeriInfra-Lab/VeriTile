import bench.tritonbench_g.cross_entropy2.Proofs.Real

namespace VeriTile.Bench.Repairs.CrossEntropy2.RealSource
open VeriTile.Triton Source
noncomputable section
set_option maxHeartbeats 6000000

theorem backward_lane (p : Parameters) (m : Memory (WithBot ℝ)) (row block : Nat) (x : Nat → ℝ)
    (input : Input p.toParameters m row x) (target : Int) (L g : ℝ)
    (labelRead : VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters m row=target)
    (lseRead : (m p.lse (p.lseBase+row)).read .real=↑L)
    (upstreamRead : (m p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑g)
    (lane : Fin p.block) (active : VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols) :
    Source.backwardValues algebra p m row block lane=
      ↑(if target=p.ignored then 0 else (g*p.logitScale)*
        ((1+2*p.squareScale*L)*Real.exp (Math.scaled p.logitScale x
          (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val)-L)-
          (if p.classStart+(VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val : Int)=target
           then 1-Math.smoothing p.smoothing p.hasSmoothing else 0)-
          Math.smoothing p.smoothing p.hasSmoothing/p.totalClasses)) := by
  by_cases ignored : target=p.ignored
  · simp [Source.backwardValues,labelRead,ignored,Source.fp,algebra,
      VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,VeriTile.Bench.Repairs.CrossEntropy1.RealSource.algebra]
  · have loaded := VeriTile.Bench.Repairs.CrossEntropy1.RealSource.loaded p.toParameters
      (scaledMemory p.logitScale m) row block (Math.scaled p.logitScale x) (scaled_input p m row x input) lane
    rw [← loaded_transfer p m row block,if_pos active] at loaded
    have lseValue : load algebra m p.lse (p.lseBase+row)=↑L := lseRead
    have upstreamValue : load algebra m p.upstream (p.upstreamBase+row*p.upstreamStride)=↑g := upstreamRead
    have indicator : (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val : Int)=target-p.classStart ↔
        p.classStart+(VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val : Int)=target := by omega
    cases hs : p.hasSmoothing
    all_goals by_cases hit : p.classStart+(VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val : Int)=target
    all_goals
      simp only [Source.backwardValues,labelRead,ignored,↓reduceIte,hs,loaded,lseValue,upstreamValue,
        indicator,hit,Bool.false_eq_true,Math.smoothing,↓reduceIte]
      simp only [Source.fp,Source.nat,Source.binary,Source.unary,
        VeriTile.Bench.Repairs.CrossEntropy1.RealSource.fp_coe,
        VeriTile.Bench.Repairs.CrossEntropy1.RealSource.nat_coe,
        VeriTile.Bench.Repairs.CrossEntropy1.RealSource.add_coe,
        VeriTile.Bench.Repairs.CrossEntropy1.RealSource.sub_coe,
        VeriTile.Bench.Repairs.CrossEntropy1.RealSource.mul_coe,
        VeriTile.Bench.Repairs.CrossEntropy1.RealSource.div_coe,
        VeriTile.Bench.Repairs.CrossEntropy1.RealSource.exp_coe,Nat.cast_one]
      apply congrArg (fun v : ℝ => (↑v : WithBot ℝ)) <;> ring

#axiomsClean backward_lane
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.RealSource
