import bench.tritonbench_g.parallel_attention.ScalarProjection
import bench.tritonbench_g.parallel_attention.ScaleForwardReal

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem forward_scalar_step {α : Type} [Inhabited α] (a : Scalars α)
    (shape : Shape) (scale : ℝ) (s : State α) (d : Bool) (j : Nat)
    (r : Fin 128) (v : Fin shape.BV) (acc : ForwardAccumulator α 128 shape.BV) :
    let c := forwardConfig shape scale
    let tile := launchTile shape s
    let start := (if d then tile.chunk*128 else 0)+j*32
    let result := forwardUpdate (scalarAlgebra a) (scalarDotPrimitive a) c s d start acc
    (result.output ⟨r,v,()⟩,result.normalizer ⟨r,()⟩) =
      forwardStep (a.ops scale) sourcePlans shape (inputReaders shape s) tile r.val v.val d j
        (acc.output ⟨r,v,()⟩,acc.normalizer ⟨r,()⟩) := by
  dsimp only
  let c := forwardConfig shape scale
  let tile := launchTile shape s
  let x := inputReaders shape s
  let start := (if d then tile.chunk*128 else 0)+j*32
  let key := matrixRead (scalarAlgebra a) s "k" (s.pids 2*(shape.T*shape.K))
    shape.K shape.T 1 shape.K (tile.key*shape.BK) start shape.BK 32
  let value := matrixRead (scalarAlgebra a) s "v" (s.pids 2*(shape.T*shape.V))
    shape.T shape.V shape.V 1 start (tile.value*shape.BV) 32 shape.BV
  let raw := scalarDot a [] (forwardQuery (scalarAlgebra a) c s) key
  let scores : TileIndex [128,32] → α := fun i =>
    let squared := a.mul (raw i) (raw i)
    if d then if start-tile.chunk*128+i.2.1.val ≤ i.1.val then squared else a.literal 0.0 else squared
  have hraw (u : Fin 32) : raw ⟨r,u,()⟩ =
      dot (a.ops scale) sourcePlans [0,0,tile.bh,tile.chunk,tile.value,r.val,j,u.val] shape.BK
        (queryLoad (a.ops scale) shape x tile 128 r.val)
        (fun e => load (a.ops scale) shape.T shape.K x.k tile.bh (start+u.val) (tile.key*shape.BK+e)) := by
    apply scalar_dot_projection
    · intro e
      simp [forwardQuery,c,forwardConfig,matrixRead,scalarAlgebra,queryChunk,keyChunk,
        queryLoad,load,Scalars.ops,x,inputReaders,tile,launchTile]
    · intro e
      simp [key,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders,tile,launchTile,and_comm,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  have hscore (u : Fin 32) : scores ⟨r,u,()⟩ =
      Math.forwardScore (a.ops scale) sourcePlans shape x tile r.val j u.val d := by
    have hs := congrArg (fun z => if d then
        if start-tile.chunk*128+u.val ≤ r.val then a.mul z z else a.literal 0.0
      else a.mul z z) (hraw u)
    have zero : a.literal (0.0 : ℝ)=a.literal 0 := congrArg a.literal (by norm_num)
    cases d with
    | «false» => simpa [scores,Math.forwardScore,Scalars.ops,start] using hs
    | «true» =>
      by_cases hc : j*32+u.val ≤ r.val
      · have hn : ¬r.val<j*32+u.val := by omega
        simpa [scores,Math.forwardScore,Scalars.ops,start,hc,hn] using hs
      · have hn : r.val<j*32+u.val := by omega
        simpa [scores,Math.forwardScore,Scalars.ops,start,hc,hn,zero] using hs
  have hn := scalar_sum_projection a scale [0,1,tile.bh,tile.chunk,tile.value,r.val,j]
    scores r (fun u => Math.forwardScore (a.ops scale) sourcePlans shape x tile r.val j u d) hscore
  have ho := scalar_dot_projection a scale [0,2,tile.bh,tile.chunk,tile.value,r.val,j,v.val]
    (fun i => a.cast (scores i)) value r v
    (fun u => a.cast (Math.forwardScore (a.ops scale) sourcePlans shape x tile r.val j u d))
    (fun u => load (a.ops scale) shape.T shape.V x.v tile.bh (start+u) (tile.value*shape.BV+v.val))
    (by intro u; exact congrArg a.cast (hscore u)) (by intro u; simp [value,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders,tile,launchTile])
  change (a.add (acc.output ⟨r,v,()⟩) (scalarDot a [] (fun i => a.cast (scores i)) value ⟨r,v,()⟩),
      a.add (acc.normalizer ⟨r,()⟩) (scalarSum a ⟨1,by simp⟩ Bool.false scores ⟨r,()⟩)) = _
  exact Prod.ext (congrArg (a.add (acc.output ⟨r,v,()⟩)) ho)
    (congrArg (a.add (acc.normalizer ⟨r,()⟩)) hn)

#axiomsClean forward_scalar_step
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
