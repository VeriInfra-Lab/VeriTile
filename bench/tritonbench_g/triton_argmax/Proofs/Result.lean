import bench.tritonbench_g.triton_argmax.Proofs.Native

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Result
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction Memory
noncomputable section

/-- Complete global byte execution returns the independent staged target.
All input/scratch capacities, disjointness and initialized index padding are
kernel-entry descriptor/scalar-memory facts. No successful-result premise. -/
theorem global_readback {M : Type} {d : MidFormat} (mem : Primitives M) (scratch : Scratch d)
    (floating : Bool) (c : GlobalConfig) (g : Native.GlobalPrograms c)
    (p : GlobalMemory.Buffers) (layout : GlobalMemory.Layout d c p) (s : M)
    (positive : 0<c.block) (midPositive : 0<c.midBlock)
    (active : ∀ pid,pid<c.blocks → pid*c.block<c.size)
    (initialized : ∀ i,i<c.midBlock → mem.read 8 s (address p.midIndex 8 i)=0) :
    mem.read 8 (Native.global mem scratch floating g p s) (address p.output 8 0)=
      BitVec.ofNat 64 (globalReference scratch c positive midPositive (GlobalMemory.input mem p s)) := by
  rw [Native.global_source mem scratch floating g p s positive active,
    GlobalMemory.global_readback mem scratch.scalars c p layout s initialized,
    global_reference scratch c positive midPositive _
      (fun pid _ => g.firstNumeric pid ▸ (g.first pid).numericComplete)
      (g.secondNumeric ▸ g.second.numericComplete)]

theorem global_frame {M : Type} {d : MidFormat} (mem : Primitives M) (scratch : Scratch d)
    (floating : Bool) (c : GlobalConfig) (g : Native.GlobalPrograms c)
    (p : GlobalMemory.Buffers) (s : M) (positive : 0<c.block)
    (active : ∀ pid,pid<c.blocks → pid*c.block<c.size) (point : Address)
    (mid : ∀ pid,pid<c.blocks → Disjoint (address p.midValue d.bytes pid) d.bytes point 1)
    (index : ∀ pid,pid<c.blocks → Disjoint (address p.midIndex 8 pid) 8 point 1)
    (out : Disjoint (address p.output 8 0) 8 point 1) :
    mem.byte (Native.global mem scratch floating g p s) point=mem.byte s point := by
  rw [Native.global_source mem scratch floating g p s positive active]
  exact GlobalMemory.global_bytes_frame mem scratch.scalars c p s point mid index out

theorem dimension_readback {M : Type} (mem : Primitives M) (c : DimConfig)
    (g : Native.DimPrograms c) (p : DimMemory.Buffers) (layout : DimMemory.Layout c p)
    (s : M) (nonempty : 0<c.extent) (row inner : Nat) (hr : row<c.outer) (hi : inner<c.inner) :
    mem.read 8 (Native.dimension mem g p s) (address p.output 8 (dimOutputOffset c row inner))=
      BitVec.ofNat 64 ((reference c.extent nonempty
        (fun i : Fin c.extent => rowInput c (DimMemory.input mem p s) row inner i.val)).val) := by
  rw [Native.dimension_source,DimMemory.dimension_readback mem fp32Scalars c p layout s row inner hr hi,
    dimension_reference c _ row inner hr nonempty
      (fun tile => g.numeric row inner tile ▸ (g.sites row inner tile).numericComplete)]

theorem dimension_frame {M : Type} (mem : Primitives M) (c : DimConfig)
    (g : Native.DimPrograms c) (p : DimMemory.Buffers) (layout : DimMemory.Layout c p)
    (s : M) (point : Address)
    (apart : ∀ row inner,row<c.outer → inner<c.inner →
      Disjoint (address p.output 8 (dimOutputOffset c row inner)) 8 point 1) :
    mem.byte (Native.dimension mem g p s) point=mem.byte s point := by
  rw [Native.dimension_source]
  exact DimMemory.dimension_bytes_frame mem fp32Scalars c p layout s point apart

#axiomsClean global_readback
#axiomsClean global_frame
#axiomsClean dimension_readback
#axiomsClean dimension_frame
#specNonCircular globalReference avoiding [Native.global,GlobalMemory.run]
#specNonCircular reference avoiding [Native.dimension,DimMemory.run]
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Result
