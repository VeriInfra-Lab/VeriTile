import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightOps

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightEval
open VeriTile.Triton RealWeightOps
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem load (h : Call) (K pid row : Nat) (r : RegionName) (u : BlockState)
    (hc : u.regs .nat [K] "col"=some (Tile.vec fun j => pid*K+j.val))
    (hm : u.regs .bool [K] "mask"=some (Tile.vec fun j => decide (pid*K+j.val<h.cols)))
    (hr : u.regs .nat [] "row"=some (Tile.scalar row)) :
    evalOp (loadOp h K r) u=some (Tile.vec fun j => some
      (if pid*K+j.val<h.cols then u.readMem r (row*h.cols+(pid*K+j.val)) else 0)) := by
  simp [loadOp,offsetOp,colOp,maskOp,evalOp,evalOp.eq_def,hc,hm,hr,Option.bind,
    Tile.bop,Tile.cop,Tile.vec,NumericDType.mul,NumericDType.add]
  funext i
  split_ifs <;> rfl

theorem inverse (row : Nat) (u : BlockState)
    (hr : u.regs .nat [] "row"=some (Tile.scalar row)) :
    evalOp (.load .real (.region "r" (.ref .nat [] "row")) .none) u=
      some (Tile.scalar (some (u.readMem "r" row))) := by
  simp [evalOp,evalOp.eq_def,hr,Option.bind]

theorem normed (K : Nat) (x : Fin K → ℝ) (inv : ℝ) (u : BlockState)
    (hx : u.regs .real [K] "x"=some (Tile.vec fun j => some (x j)))
    (hr : u.regs .real [] "inv"=some (Tile.scalar (some inv))) :
    evalOp (.mul .real .scalarR (refOp K "x") (.ref .real [] "inv")) u=
      some (Tile.vec fun j => some (x j*inv)) := by
  simp [refOp,evalOp,evalOp.eq_def,hx,hr,Option.bind,Tile.bop,Tile.vec,NumericDType.mul]

theorem accumulation (K : Nat) (acc dy normed : Fin K → ℝ) (u : BlockState)
    (ha : u.regs .real [K] "acc"=some (Tile.vec fun j => some (acc j)))
    (hd : u.regs .real [K] "dy"=some (Tile.vec fun j => some (dy j)))
    (hn : u.regs .real [K] "normed"=some (Tile.vec fun j => some (normed j))) :
    evalOp (.add .real (.consSame .nil) (refOp K "acc")
      (.mul .real (.consSame .nil) (refOp K "dy") (refOp K "normed"))) u=
      some (Tile.vec fun j => some (acc j+dy j*normed j)) := by
  simp [refOp,evalOp,evalOp.eq_def,ha,hd,hn,Option.bind,Tile.bop,Tile.vec,NumericDType.mul,NumericDType.add]

#axiomsClean load
#axiomsClean inverse
#axiomsClean normed
#axiomsClean accumulation
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightEval
