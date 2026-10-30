import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ScalarProjection
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ScaleKeyValueReal

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem keyValue_scalar_step {α : Type} [Inhabited α] (a : Scalars α)
    (shape : Shape) (scale : ℝ) (tile : Tile) (s : State α) (d : Bool) (j : Nat)
    (r : Fin 64) (e : Fin shape.BK) (v : Fin shape.BV)
    (acc : BackwardAccumulator α 64 shape.BK shape.BV) :
    let result := backwardUpdate (scalarAlgebra a) (scalarDotPrimitive a) (backwardConfig shape scale tile)
      s d (Math.kvStart shape tile d j) acc
    (result.key ⟨r,e,()⟩,result.value ⟨r,v,()⟩) =
      keyValueStep (a.ops scale) sourcePlans shape (inputReaders shape s) tile r.val e.val v.val d j
        (acc.key ⟨r,e,()⟩,acc.value ⟨r,v,()⟩) := by
  dsimp only
  let c := backwardConfig shape scale tile
  let x := inputReaders shape s
  let start := Math.kvStart shape tile d j
  let query := matrixRead (scalarAlgebra a) s "q" (tile.bh*(shape.T*shape.K))
    shape.K shape.T 1 shape.K (tile.key*shape.BK) start shape.BK 32
  let outGrad := fun i => a.cast (rawMatrixRead (scalarAlgebra a) s "do" (tile.bh*(shape.T*shape.V))
    shape.V shape.T 1 shape.V (tile.value*shape.BV) start shape.BV 32 i)
  let normGrad := fun (u : Fin 32) => if start+u.val < shape.T then
    (s.mem "dz" (tile.bh*shape.T+start+u.val)).read .real else a.literal 0.0
  let key := if d then backwardKey (scalarAlgebra a) c s
    else fun i => a.cast (backwardKey (scalarAlgebra a) c s i)
  let scaled := fun (i : TileIndex [64,32]) => a.mul (scalarDot a [] key query i) (a.literal scale)
  let valid := fun (i : TileIndex [64,32]) => start+i.2.1.val < shape.T
  let mask := fun (i : TileIndex [64,32]) => i.1.val ≤ start-tile.chunk*64+i.2.1.val ∧ valid i
  let score := fun i => if d then if mask i then scaled i else a.literal 0.0
    else if valid i then scaled i else a.literal 0.0
  let square := fun i => if d then if mask i then a.mul (scaled i) (scaled i) else a.literal 0.0
    else a.mul (score i) (score i)
  let rawDS := scalarDot a [] (backwardValue (scalarAlgebra a) c s) outGrad
  let ds := fun i => if d then
      a.mul (if mask i then (if tile.value=0 then a.add (rawDS i) (normGrad i.2.1) else rawDS i)
        else a.literal 0.0) (a.literal scale)
    else if valid i then
      if tile.value=0 then a.add (a.mul (rawDS i) (a.literal scale)) (a.mul (normGrad i.2.1) (a.literal scale))
      else a.mul (rawDS i) (a.literal scale)
    else a.literal 0.0
  have zero : a.literal (0.0 : ℝ)=a.literal 0 := congrArg a.literal (by norm_num)
  have two : a.literal (2.0 : ℝ)=a.literal 2 := congrArg a.literal (by norm_num)
  have hraw (u : Fin 32) : scaled ⟨r,u,()⟩ = Math.kvRaw (a.ops scale) sourcePlans shape x tile r.val j u.val d := by
    apply congrArg (fun z => a.mul z (a.literal scale))
    apply scalar_dot_projection
    · intro f
      cases d <;> simp [key,backwardKey,c,backwardConfig,forwardConfig,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders]
    · intro f
      simp [query,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders,start,and_comm,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  have hscore (u : Fin 32) : score ⟨r,u,()⟩ = Math.kvScore (a.ops scale) sourcePlans shape x tile r.val j u.val d := by
    have h := congrArg (fun z => if d then if mask ⟨r,u,()⟩ then z else a.literal 0.0
      else if valid ⟨r,u,()⟩ then z else a.literal 0.0) (hraw u)
    cases d <;> simpa [score,Math.kvScore,Math.kvValid,Scalars.ops,mask,valid,start,Math.kvStart,zero,and_comm] using h
  have hsquare (u : Fin 32) : square ⟨r,u,()⟩ = Math.kvSquare (a.ops scale) sourcePlans shape x tile r.val j u.val d := by
    cases d with
    | «false» => exact congrArg₂ a.mul (hscore u) (hscore u)
    | «true» =>
      have h := congrArg (fun z => if mask ⟨r,u,()⟩ then a.mul z z else a.literal 0.0) (hraw u)
      simpa [square,Math.kvSquare,Math.kvValid,Scalars.ops,mask,valid,start,Math.kvStart,zero,and_comm] using h
  have hrawDS (u : Fin 32) : rawDS ⟨r,u,()⟩ =
      dot (a.ops scale) sourcePlans [2,1,tile.bh,tile.chunk,tile.value,r.val,j,u.val] shape.BV
        (fun f => load (a.ops scale) shape.T shape.V x.v tile.bh (tile.chunk*64+r.val) (tile.value*shape.BV+f))
        (fun f => a.cast (load (a.ops scale) shape.T shape.V x.do_ tile.bh (start+u.val) (tile.value*shape.BV+f))) := by
    apply scalar_dot_projection
    · intro f
      simp [backwardValue,c,backwardConfig,forwardConfig,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders]
    · intro f
      simp [outGrad,rawMatrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders,and_comm,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  have hz (u : Fin 32) : normGrad u = dzLoad (a.ops scale) shape x tile.bh (start+u.val) := by
    simp [normGrad,dzLoad,Scalars.ops,x,inputReaders,zero,Nat.add_assoc]
  have hds (u : Fin 32) : ds ⟨r,u,()⟩ = Math.kvDS (a.ops scale) sourcePlans shape x tile r.val j u.val d := by
    let combine := fun raw => if d then
        a.mul (if mask ⟨r,u,()⟩ then (if tile.value=0 then a.add raw (normGrad u) else raw)
          else a.literal 0.0) (a.literal scale)
      else if valid ⟨r,u,()⟩ then
        if tile.value=0 then a.add (a.mul raw (a.literal scale)) (a.mul (normGrad u) (a.literal scale))
        else a.mul raw (a.literal scale)
      else a.literal 0.0
    have h := congrArg combine (hrawDS u)
    cases d <;> simpa [ds,combine,Math.kvDS,Math.kvValid,Scalars.ops,mask,valid,start,Math.kvStart,zero,hz,and_comm] using h
  let terms := fun i => a.cast (a.mul (a.mul (a.literal 2.0) (ds i)) (score i))
  have hk := scalar_dot_projection a scale [2,3,tile.bh,tile.chunk,tile.value,r.val,j,e.val]
    terms (matrixTranspose query) r e
    (fun u => a.cast (a.mul (a.mul (a.literal 2) (Math.kvDS (a.ops scale) sourcePlans shape x tile r.val j u d))
      (Math.kvScore (a.ops scale) sourcePlans shape x tile r.val j u d)))
    (fun u => load (a.ops scale) shape.T shape.K x.q tile.bh (start+u) (tile.key*shape.BK+e.val))
    (by intro u; simpa [terms,two] using congrArg₂ (fun ds sc => a.cast (a.mul (a.mul (a.literal 2) ds) sc)) (hds u) (hscore u))
    (by intro u; simp [query,matrixTranspose,matrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders,and_comm,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm])
  have hv := scalar_dot_projection a scale [2,2,tile.bh,tile.chunk,tile.value,r.val,j,v.val]
    (fun i => a.cast (square i)) (matrixTranspose outGrad) r v
    (fun u => a.cast (Math.kvSquare (a.ops scale) sourcePlans shape x tile r.val j u d))
    (fun u => a.cast (load (a.ops scale) shape.T shape.V x.do_ tile.bh (start+u) (tile.value*shape.BV+v.val)))
    (by intro u; exact congrArg a.cast (hsquare u))
    (by intro u; simp [outGrad,matrixTranspose,rawMatrixRead,scalarAlgebra,load,Scalars.ops,x,inputReaders,and_comm,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm])
  change (a.add (acc.key ⟨r,e,()⟩) (scalarDot a [] terms (matrixTranspose query) ⟨r,e,()⟩),
    a.add (acc.value ⟨r,v,()⟩) (scalarDot a [] (fun i => a.cast (square i)) (matrixTranspose outGrad) ⟨r,v,()⟩)) = _
  exact Prod.ext (congrArg (a.add (acc.key ⟨r,e,()⟩)) hk)
    (congrArg (a.add (acc.value ⟨r,v,()⟩)) hv)

#axiomsClean keyValue_scalar_step
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
