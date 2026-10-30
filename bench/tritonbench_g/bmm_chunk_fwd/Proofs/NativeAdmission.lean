import bench.tritonbench_g.bmm_chunk_fwd.Proofs.DirectLayout
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeProgram

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Direct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 3000000

def buffer (p : Metadata) (h : InputContract p) (a : Allocation p) (r : Role) : NativePointers.Buffer :=
  if used : Used p r then
    ⟨base p r,bytes r,(p.tensor r).numel,bytes r*(p.tensor r).numel,
      by cases r <;> decide,le_rfl,h.extents.1 r used,a.inputsFit r used⟩
  else ⟨0,bytes r,0,0,by cases r <;> decide,by simp,by norm_num,by norm_num⟩

def outputBuffer (p : Metadata) (h : InputContract p) (a : Allocation p) : NativePointers.Buffer :=
  ⟨a.pointer,4,(shape p).outputSize,4*(shape p).outputSize,by decide,le_rfl,h.extents.2,a.outputFit⟩

def nativeLayout (p : Metadata) (h : InputContract p) (a : Allocation p) : NativePointers.Layout (call p) where
  a := buffer p h a .a
  b := buffer p h a .b
  seq := buffer p h a .seq
  out := outputBuffer p h a
  aWidth := by simp [buffer,Used,bytes]
  bWidth := by simp [buffer,Used,bytes]
  seqWidth := Or.inr (by simp [buffer,bytes]; split <;> rfl)
  outWidth := Or.inr (Or.inr (Or.inl rfl))
  aBounds := by
    intro b r g k hb hr hg hk
    change Host.inputOffset (Host.prepared (shape p) (p.strides .a)) b r g k<_
    rw [a_stride p h b r g k hb hr hg hk,input_offset (shape p) b r g k hg]
    simpa [buffer,Used] using input_bound p h b r g k hb hr hg hk
  bBounds := by
    intro b r g k hb hr hg hk
    change Host.inputOffset (Host.prepared (shape p) (p.strides .b)) b r g k<_
    rw [b_stride p h b r g k hb hr hg hk,input_offset (shape p) b r g k hg]
    have same : (p.tensor .b).numel=(p.tensor .a).numel := by simp [VeriTile.Torch.TensorMetadata.numel,h.shapes.2]
    simpa [buffer,Used,same] using input_bound p h b r g k hb hr hg hk
  seqBounds := by
    intro seq b r hb hr
    have hs : p.hasSeq=Bool.true := seq
    have size : (p.tensor .seq).numel=(shape p).batch*(shape p).seqlen := by
      unfold VeriTile.Torch.TensorMetadata.numel
      rw [seq_shape p h hs]; simp
    simpa [buffer,Used,hs,size,call,(seq_stride p h hs b r hb hr).1,
      (seq_stride p h hs b r hb hr).2] using Host.mul_add_bound hb hr
  outExtent := rfl

theorem nativeMetadata (p : Metadata) (h : InputContract p) (c : Host.Config)
    (hc : c∈Host.configurations) (nonempty : 0<(shape p).outputSize) : NativeControl.Metadata (call p) c := by
  have grid := h.grid (by omega)
  refine ⟨finite_sizes p h nonempty,?_,?_,?_⟩
  · have := grid.2.2 c hc
    change Host.cdiv (shape p).chunk c.M*Host.cdiv (shape p).chunk c.N≤2147483647
    norm_num at this
    omega
  · change (shape p).batch≤65535
    have := grid.1
    norm_num at this
    omega
  · change (shape p).chunks*(shape p).groups≤65535
    have := grid.2.1
    norm_num at this
    omega

theorem nativeCertificate {α : Type} [Inhabited α] (ops : FPSuccess.Ops α)
    (p : Metadata) (h : InputContract p) (a : Allocation p) (c : Host.Config)
    (hc : c∈Host.configurations) (nonempty : 0<(shape p).outputSize) :
    NativeProgram.Certificate ops (call p) c (nativeLayout p h a) :=
  NativeProgram.certify ops (call p) c hc (nativeMetadata p h c hc nonempty) (nativeLayout p h a) rfl rfl

#axiomsClean nativeCertificate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Direct
