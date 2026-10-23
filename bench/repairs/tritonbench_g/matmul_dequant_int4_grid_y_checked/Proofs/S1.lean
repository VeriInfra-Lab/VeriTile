import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.TensorResults

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.S1
open VeriTile.Triton FP.Structural Tensor
open MatmulDequantInt4.Direct
noncomputable section

theorem packed_one (p : PackedInfo) (m : Nat) (f : Source.Profile)
    (h : PackedContract p m f.hostDtype) : PackedContract p 1 f.hostDtype := by
  rcases h with ⟨⟨integer,pack,flags⟩,axisBound⟩
  rcases pack with ⟨selected,shape,weight,scale,zero,independent,indices⟩
  refine ⟨⟨integer,⟨selected,?_,weight,scale,zero,independent,⟨indices.weight,indices.scale,indices.zero,indices.output,
    indices.grid,indices.k,indices.n,indices.group,indices.packed⟩⟩,flags⟩,axisBound⟩
  exact ⟨by change 0<1; decide,shape.npos,shape.kpos,shape.gpos,shape.k8,shape.n8,shape.groups⟩

/-- Production s1 executes this complete FP16 dequantizer. Its ordinary
Torch quantizer and torch.mm retain the existing library specifications. -/
theorem kernel_real (p : S1Info) (h : S1Contract p) (o : Info)
    (t : Arguments p.packedInfo o) (output : OutputABI p.packedInfo .half o)
    (i : Fin 8) (heap : Heap ℝ) :
    ∃ final,KernelGrid.run .half Spec.real (params p.packedInfo i) (entry .half p.packedInfo o t heap)=some final ∧
      Post p.packedInfo o t (Spec.mathematical p.packedInfo o t heap) heap
        (physical .half p.packedInfo o t heap final) := by
  exact TensorBridge.real .half p.packedInfo o t
    ((checkPacked_iff _ _ _).mpr (packed_one p.packedInfo p.input.tensor.rows .half ⟨h.1.1,h.2⟩)) output i heap

theorem kernel_native (p : S1Info) (h : S1Contract p) (o : Info)
    (t : Arguments p.packedInfo o) (output : OutputABI p.packedInfo .half o)
    (I : Native.Instructions) (conversion : Native.Conversion I) (i : Fin 8) (heap : Heap Native.Word) :
    ∃ final,KernelGrid.run .half (Native.scalar I) (params p.packedInfo i) (entry .half p.packedInfo o t heap)=some final ∧
      Post p.packedInfo o t (TensorBridge.target .half I p.packedInfo o t heap) heap
        (physical .half p.packedInfo o t heap final) := by
  exact TensorBridge.native .half I (fun _ => conversion) p.packedInfo o t
    ((checkPacked_iff _ _ _).mpr (packed_one p.packedInfo p.input.tensor.rows .half ⟨h.1.1,h.2⟩)) output i heap

#axiomsClean packed_one
#axiomsClean kernel_real
#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.S1
