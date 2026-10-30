import bench.tritonbench_g.token_softmax_bloom.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOScalar
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open VeriTile.Bench.TritonBenchG.SoftmaxTriton1 (softmaxSpec softmaxInputTile softmaxSpec_eq_formula)
open SoftmaxIO
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem real (p : Parameters) (positive : 0 < p.width) (work : List Grid.PID)
    (vals : (realIO p work).Inputs) (s : State Real.Scalar)
    (binding : (realIO p work).InputBinding vals s) (r : Fin work.length)
    (i : Fin p.width) (active : i.val < SoftmaxIO.length p work vals r) :
    WithBot.unbotD 0 (outputValues Real.algebra .fp32 p (Grid.state s.mem (work.get r)) (i,())) =
      realRow (SoftmaxIO.length p work vals r) (fun c => vals (2 : Fin 3) (r,c,())) i := by
  have hc := IOBinding.length_read Real.algebra (fun x : ℝ => some x) p
    (fun _ => WithBot.unbotD 0) work vals s binding r
  have hs := IOBinding.start_read Real.algebra (fun x : ℝ => some x) p
    (fun _ => WithBot.unbotD 0) work vals s binding r
  have rows : inputRow Real.algebra .fp32 p (Grid.state s.mem (work.get r)) =
      fun j => if j.1.val < SoftmaxIO.length p work vals r then some (vals (2 : Fin 3) (r,j.1,())) else none := by
    funext j
    unfold inputRow
    rw [hc,hs]
    by_cases h : j.1.val < SoftmaxIO.length p work vals r
    · simpa only [Real.algebra, if_pos h, toFloat] using
        IOBinding.payload_read Real.algebra (fun x : ℝ => some x) p
          (fun _ => WithBot.unbotD 0) work vals s binding r j.1 h
    · simp only [Real.algebra, if_neg h]
      rfl
  have observed : WithBot.unbotD 0
      (outputValues Real.algebra .fp32 p (Grid.state s.mem (work.get r)) (i,())) =
      softmaxSpec (SoftmaxIO.length p work vals r) p.width (fun c => vals (2 : Fin 3) (r,c,())) i := by
    simp only [outputValues,numerators,rows]
    simp [Real.algebra,softmaxSpec,softmaxInputTile,Tile.reduceMax,Tile.reduceMaxDrop,
      Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.eraseAxis,
      TileShape.insertAxisIndex,Tile.bop,Tile.uop,positive]
    rfl
  rw [observed,softmaxSpec_eq_formula _ _ positive _ i active]
  rfl

theorem native (I : Native.Instructions) (V : Nat) (p : Parameters) (work : List Grid.PID)
    (vals : (nativeIO I V p work).Inputs) (s : State Native.Node)
    (binding : (nativeIO I V p work).InputBinding vals s) (r : Fin work.length)
    (i : Fin p.width) :
    outputValues (SoftmaxTriton1.NativeGraph.algebra I V) .fp32 p
      (Grid.state s.mem (work.get r)) (i,()) (SoftmaxTriton1.NativeGraph.owner V p.width i.val) =
      nativeRow I V p.width (SoftmaxIO.length p work vals r) (fun c => vals (2 : Fin 3) (r,c,())) i := by
  let n := SoftmaxIO.length p work vals r
  let xs : Nat → Native.Word := fun j => if h : j < p.width then vals (2 : Fin 3) (r,⟨j,h⟩,()) else default
  have hc := IOBinding.length_read (SoftmaxTriton1.NativeGraph.algebra I V)
    SoftmaxTriton1.NativeGraph.atom p
    (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val)) work vals s binding r
  have hs := IOBinding.start_read (SoftmaxTriton1.NativeGraph.algebra I V)
    SoftmaxTriton1.NativeGraph.atom p
    (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val)) work vals s binding r
  have rows : inputRow (SoftmaxTriton1.NativeGraph.algebra I V) .fp32 p (Grid.state s.mem (work.get r)) =
      fun j => if j.1.val < n then SoftmaxTriton1.NativeGraph.atom (xs j.1.val)
        else SoftmaxTriton1.NativeGraph.atom SoftmaxTriton1.NativeGraph.negInf := by
    funext j
    unfold inputRow
    rw [hc,hs]
    by_cases h : j.1.val < n
    · simpa only [SoftmaxTriton1.NativeGraph.algebra, n, if_pos h, toFloat, xs, dif_pos j.1.isLt] using
        IOBinding.payload_read (SoftmaxTriton1.NativeGraph.algebra I V)
          SoftmaxTriton1.NativeGraph.atom p
          (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val)) work vals s binding r j.1 h
    · simp only [SoftmaxTriton1.NativeGraph.algebra, n, if_neg h]
  have lowered := SoftmaxTriton1.NativeGraph.source_graph_lowering I V n p.width xs i.val i.isLt
  change _ = SoftmaxTriton1.NativeGraph.row I V n p.width xs i.val
  rw [←lowered]
  simp only [outputValues,numerators,rows,SoftmaxTriton1.Floating.formula,
    dif_pos i.isLt,SoftmaxTriton1.Floating.rowExpression]
  rfl

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOScalar
