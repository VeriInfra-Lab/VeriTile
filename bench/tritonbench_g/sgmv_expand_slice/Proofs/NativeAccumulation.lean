import bench.tritonbench_g.sgmv_expand_slice.Proofs.NativeOperands
import bench.tritonbench_g.sgmv_expand_slice.Proofs.FoldValues

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 20000

theorem update_matches (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (input : Native.Inputs (profile abi)) (s : State Raw)
    (memory : SourceMemory.Inputs abi p shape s (inputWords abi input) (weightWords abi input))
    (fits : SegmentFits shape segment) (live : segment.active) (positive : 0 < shape.columns)
    (pid block : Nat) (hb : block < (p.inner+15)/16)
    (acc : Values Raw .real [32,32]) (reference : Native.Matrix .single 32 32)
    (same : ∀ r c, acc (r,c,PUnit.unit) = reference r c) (i : TileIndex [32,32]) :
    SourceBody.update (algebra rest ops (profile abi)) (dot ops (weight abi.weight)) abi p
      (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) s block acc i =
      Native.mmaTile ops (weight abi.weight)
        (Native.aTile ops (profile abi) shape segment input (pid/tiles p.columns) block)
        (Native.bTile (profile abi) shape segment input (pid%tiles p.columns) block) reference i.1 i.2.1 := by
  have ha : (fun r k => unpack (weight abi.weight).format
      (SourceBody.aValues (algebra rest ops (profile abi)) abi p
        (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) s block
          (r,k,PUnit.unit))) =
      Native.aTile ops (profile abi) shape segment input (pid/tiles p.columns) block := by
    funext r k
    rw [a_matches rest ops zeros abi p shape segment input s memory fits live pid block hb,unpack_pack]
  have hb' : (fun k c => unpack (weight abi.weight).format
      (SourceBody.bValues (algebra rest ops (profile abi)) abi p
        (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) s block
          (k,c,PUnit.unit))) =
      Native.bTile (profile abi) shape segment input (pid%tiles p.columns) block := by
    funext k c
    rw [b_matches rest ops zeros abi p shape segment input s memory fits live positive pid block hb,unpack_pack]
  have hc : (fun r c => acc (r,c,PUnit.unit)) = reference := by funext r c; exact same r c
  simp only [SourceBody.update,selected_dot,ha,hb',hc]

theorem accumulated_matches (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (input : Native.Inputs (profile abi)) (s : State Raw)
    (memory : SourceMemory.Inputs abi p shape s (inputWords abi input) (weightWords abi input))
    (fits : SegmentFits shape segment) (live : segment.active) (positive : 0 < shape.columns)
    (pid : Nat) (i : TileIndex [32,32]) :
    SourceBody.accumulated (algebra rest ops (profile abi)) (dot ops (weight abi.weight)) abi p
      (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) s 0
      (SourceInit.zero (algebra rest ops (profile abi))) i =
      Native.accumulate ops (profile abi) shape segment input (pid/tiles p.columns) (pid%tiles p.columns)
        i.1 i.2.1 := by
  have h : ∀ r c,
      SourceBody.accumulated (algebra rest ops (profile abi)) (dot ops (weight abi.weight)) abi p
        (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) s 0
        (SourceInit.zero (algebra rest ops (profile abi))) (r,c,PUnit.unit) =
        Native.accumulate ops (profile abi) shape segment input (pid/tiles p.columns) (pid%tiles p.columns) r c := by
    unfold SourceBody.accumulated Native.accumulate
    rw [foldRange_range,←memory.inner]
    apply List.foldl_rel (r := fun (a : Values Raw .real [32,32]) (b : Native.Matrix .single 32 32) =>
      ∀ r c, a (r,c,PUnit.unit) = b r c)
    · intro r c; rfl
    · intro block hb acc reference same r c
      exact update_matches rest ops zeros abi p shape segment input s memory fits live positive
        pid block (List.mem_range.mp hb) acc reference same (r,c,PUnit.unit)
  rcases i with ⟨r,c,u⟩
  cases u
  exact h r c

#axiomsClean update_matches
#axiomsClean accumulated_matches
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
