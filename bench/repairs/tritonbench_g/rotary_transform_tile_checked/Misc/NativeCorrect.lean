import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.SourceExports
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.IOResults

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section

/-- The same complete FP32 source and finite IO windows. Inputs retain all
Native.Node values; observation records their exact raw FP32 words. Metadata
still uses the source's signed integer load channels. -/
def nativeIO (c : Config) (p : Native.Instructions) (cumulative64 offsets64 : Bool) :
    ReadProgramIO Native.Node Native.Node where
  inputs := [⟨.real,[PublicIO.dataSize c.shape],"x"⟩,
    ⟨.real,[PublicIO.coefficientSize c.shape],"cos"⟩,
    ⟨.real,[PublicIO.coefficientSize c.shape],"sin"⟩,
    ⟨.int,[c.shape.batches+1],"cu"⟩,⟨.int,[c.shape.batches],"offsets"⟩,
    ⟨.real,[PublicIO.dataSize c.shape],TypedExecution.outputRegion c.shape⟩]
  outputs := [⟨.real,[c.shape.batches,c.shape.maximum,c.shape.heads,c.shape.width],
    TypedExecution.outputRegion c.shape⟩]
  scratch := []
  buffers := ["x","cos","sin","cu","offsets",TypedExecution.outputRegion c.shape]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ => fun lane => lane.1.val
    | ⟨n+6,h⟩ => False.elim (by simp at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun _ => True
    | ⟨3,_⟩ => fun _ => c.shape.varlen = Bool.true
    | ⟨4,_⟩ => fun _ => c.shape.offsetTensor = Bool.true
    | ⟨5,_⟩ => fun lane => 2*c.shape.half ≤ lane.1.val%c.shape.width
    | ⟨n+6,h⟩ => False.elim (by simp at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => c.shape.address (c.start lane.1.val+lane.2.1.val)
      lane.2.2.1.val lane.2.2.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.2.1.val<c.length lane.1.val ∧
      lane.2.2.2.1.val<2*c.shape.half
    | ⟨n+1,h⟩ => False.elim (by simp at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => if c.shape.total=0 ∨ c.shape.half=0 then [] else
    (programs c).map fun pid => {
      kernel := rotary_transform_source c.shape cumulative64 offsets64
      backend := KernelAlgebra.backend (Native.ops p)
      pids := fun ax => if ax=0 then pid.1 else if ax=1 then pid.2.1 else pid.2.2
      numPids := some (fun ax => if ax=0 then c.shape.rowBlocks else
        if ax=1 then c.shape.batches else c.shape.heads) }
  encode := id
  observe := fun _ _ node => Native.atom node.value
  resultRead := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => c.shape.address (c.start lane.1.val+lane.2.1.val)
      lane.2.2.1.val lane.2.2.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp at h)
  resultMask := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.2.1.val<c.length lane.1.val
    | ⟨n+1,h⟩ => False.elim (by simp at h)


/-- Independent native instruction graph: low uses fma(co,a,neg(mul(sn,b)));
split and interleaved high preserve their original FMA operand ordering.
The instruction callbacks and this fixed fusion pattern are the existing
primitive/compiler model boundaries, not numerical transformation assumptions. -/
def nativeRotation (c : Config) (p : Native.Instructions) (cu64 off64 : Bool) :
    (nativeIO c p cu64 off64).Inputs → (nativeIO c p cu64 off64).Outputs :=
  PublicIO.expected c (fun values batch row head col => Native.atom (
    let m := Native.project values
    if col<2*c.shape.half then
      let pair := c.shape.pairOf col
      let second := c.shape.second col
      let token := c.start batch+row
      let a := m .input (c.shape.address token head (c.shape.dim pair Bool.false))
      let b := m .input (c.shape.address token head (c.shape.dim pair Bool.true))
      let cache := row+c.shift batch<c.shape.coefficientRows
      let co := if cache then m .cosine ((row+c.shift batch)*c.shape.half+pair) else Native.one
      let sn := if cache then m .sine ((row+c.shift batch)*c.shape.half+pair) else Native.zero
      let signed := if c.shape.conjugate then p.sub0 sn else sn
      if second then
        if c.shape.interleaved then p.fma co b (p.mul signed a) else p.fma signed a (p.mul co b)
      else p.fma co a (p.neg (p.mul signed b))
    else m .input (c.shape.address (c.start batch+row) head col)))

/-- Exact native readback of the selected complete grid, including exceptional
FP32 payloads, the unchanged suffix and every exterior tagged cell. -/
theorem rotary_transform_native (c : Config) (valid : c.Valid) (p : Native.Instructions)
    (cu64 off64 : Bool) :
    (nativeIO c p cu64 off64).ImplementsOn (PublicIO.domain c) (nativeRotation c p cu64 off64) :=
  IOResults.native c valid p cu64 off64

#specNonCircular nativeRotation avoiding [TypedGrid.run,ProgramModel.result,Native.graph,Native.pairWord,Native.ops]
#axiomsClean rotary_transform_native
#stmtSurfaceSubset rotary_transform_native ⊆ [nativeIO,nativeRotation,ReadProgramIO.ImplementsOn,
  PublicIO.domain,Config,Config.Valid,Native.Instructions,Native.Node,
  KernelEndpoints.instInhabitedNode,ReadProgramIO.Inputs,ReadProgramIO.Outputs]
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked
