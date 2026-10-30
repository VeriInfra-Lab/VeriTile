import bench.tritonbench_g.parallel_attention.ScalarProjection
import bench.tritonbench_g.parallel_attention.ScaleQueryReal

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem query_scalar_step {α : Type} [Inhabited α] (a : Scalars α)
    (shape : Shape) (scale : ℝ) (tile : Tile) (s : State α) (d : Bool) (j : Nat)
    (r : Fin 64) (e : Fin shape.BK) (acc : TileIndex [64,shape.BK] → α) :
    queryUpdate (scalarAlgebra a) (scalarDotPrimitive a) (dqConfig shape scale tile) s d
      ((if d then tile.chunk*64 else 0)+j*32) acc ⟨r,e,()⟩ =
      queryStep (a.ops scale) sourcePlans shape (inputReaders shape s) tile r.val e.val d j (acc ⟨r,e,()⟩) := by
  let c := dqConfig shape scale tile
  let x := inputReaders shape s
  let start := (if d then tile.chunk*64 else 0)+j*32
  let key := matrixRead (scalarAlgebra a) s "k" (tile.bh*(shape.T*shape.K))
    shape.T shape.K shape.K 1 start (tile.key*shape.BK) 32 shape.BK
  let value := matrixRead (scalarAlgebra a) s "v" (tile.bh*(shape.T*shape.V))
    shape.V shape.T 1 shape.V (tile.value*shape.BV) start shape.BV 32
  let rawDS := scalarDot a [] (queryOutGrad (scalarAlgebra a) c s) value
  let rawScore := scalarDot a [] (queryInput (scalarAlgebra a) c s) (matrixTranspose key)
  let terms : TileIndex [64,32] → α := fun i =>
    let ds0 := if tile.value=0 then a.add (rawDS i) (queryNormGrad (scalarAlgebra a) c s ⟨i.1,()⟩) else rawDS i
    let mask := start-tile.chunk*64+i.2.1.val ≤ i.1.val
    let ds := if d then a.mul (if mask then ds0 else a.literal 0.0) (a.literal scale) else ds0
    let score := if d then if mask then rawScore i else a.literal 0.0 else rawScore i
    a.cast (a.mul (a.mul (a.literal 2.0) ds) score)
  have hds (u : Fin 32) : rawDS ⟨r,u,()⟩ =
      dot (a.ops scale) sourcePlans [1,0,tile.bh,tile.chunk,tile.value,r.val,j,u.val] shape.BV
        (fun f => a.cast (load (a.ops scale) shape.T shape.V x.do_ tile.bh (tile.chunk*64+r.val) (tile.value*shape.BV+f)))
        (fun f => load (a.ops scale) shape.T shape.V x.v tile.bh (start+u.val) (tile.value*shape.BV+f)) := by
    apply scalar_dot_projection
    · intro f
      simp [queryOutGrad,c,dqConfig,backwardConfig,forwardConfig,rawMatrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders]
    · intro f
      simp [value,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders,and_comm,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  have hsc (u : Fin 32) : rawScore ⟨r,u,()⟩ =
      dot (a.ops scale) sourcePlans [1,1,tile.bh,tile.chunk,tile.value,r.val,j,u.val] shape.BK
        (queryLoad (a.ops scale) shape x tile 64 r.val)
        (fun f => load (a.ops scale) shape.T shape.K x.k tile.bh (start+u.val) (tile.key*shape.BK+f)) := by
    apply scalar_dot_projection
    · intro f
      simp [queryInput,c,dqConfig,backwardConfig,forwardConfig,matrixRead,scalarAlgebra,
        queryLoad,load,Scalars.ops,x,inputReaders]
    · intro f
      simp [matrixTranspose,key,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders]
  have hz : queryNormGrad (scalarAlgebra a) c s ⟨r,()⟩ =
      dzLoad (a.ops scale) shape x tile.bh (tile.chunk*64+r.val) := by
    norm_num [queryNormGrad,c,dqConfig,backwardConfig,forwardConfig,scalarAlgebra,dzLoad,Scalars.ops,x,inputReaders,Nat.add_assoc]
  have ht (u : Fin 32) : terms ⟨r,u,()⟩ = Math.queryTerm (a.ops scale) sourcePlans shape x tile r.val j u.val d := by
    let combine := fun ds0 sc =>
      let ds1 := if tile.value=0 then a.add ds0 (dzLoad (a.ops scale) shape x tile.bh (tile.chunk*64+r.val)) else ds0
      let mask := start-tile.chunk*64+u.val ≤ r.val
      let ds := if d then a.mul (if mask then ds1 else a.literal 0.0) (a.literal scale) else ds1
      let score := if d then if mask then sc else a.literal 0.0 else sc
      a.cast (a.mul (a.mul (a.literal 2.0) ds) score)
    have hh := congrArg₂ combine (hds u) (hsc u)
    have zero : a.literal (0.0 : ℝ)=a.literal 0 := congrArg a.literal (by norm_num)
    have two : a.literal (2.0 : ℝ)=a.literal 2 := congrArg a.literal (by norm_num)
    cases d with
    | «false» => simpa [terms,combine,Math.queryTerm,Scalars.ops,start,hz,two] using hh
    | «true» =>
      by_cases hc : j*32+u.val ≤ r.val
      · have hn : ¬r.val<j*32+u.val := by omega
        simpa [terms,combine,Math.queryTerm,Scalars.ops,start,hz,hc,hn,zero,two] using hh
      · have hn : r.val<j*32+u.val := by omega
        simpa [terms,combine,Math.queryTerm,Scalars.ops,start,hz,hc,hn,zero,two] using hh
  have ho := scalar_dot_projection a scale [1,2,tile.bh,tile.chunk,tile.value,r.val,j,e.val]
    terms key r e (fun u => Math.queryTerm (a.ops scale) sourcePlans shape x tile r.val j u d)
    (fun u => load (a.ops scale) shape.T shape.K x.k tile.bh (start+u) (tile.key*shape.BK+e.val)) ht
    (by intro u; simp [key,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders])
  change a.add (acc ⟨r,e,()⟩) (scalarDot a [] terms key ⟨r,e,()⟩) = _
  exact congrArg (a.add (acc ⟨r,e,()⟩)) ho

#axiomsClean query_scalar_step
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
