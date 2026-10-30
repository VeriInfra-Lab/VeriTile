import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.IO
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.LogicalAgreement

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
open VeriTile Triton AttnFwdTriton Verified
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem agreement (A : FlatAlloc) (p : Parameters) (values : Inputs p)
    (s : BlockState) (bound : (programIO A p).InputBinding s values) :
    LogicalAgreement.Agree p
      (LogicalInputs.inputs s (Relocation.parameters p A.flat) (Relocation.offsets A p))
      (logicalInputs p values) := by
  constructor
  · intro z hz h hh n hn d hd
    have hb := bound (0:Fin 5) (coordinates p (⟨z,hz⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩)) trivial
    simpa [programIO,AlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,
      offset,inputRegion,LogicalInputs.inputs,logicalInputs,tensor,hz,hh,hn,hd,
      Relocation.parameters,Relocation.offsets,ChanTy.read] using hb
  · intro z hz h hh n hn d hd
    have hb := bound (1:Fin 5) (coordinates p (⟨z,hz⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩)) trivial
    simpa [programIO,AlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,
      offset,inputRegion,LogicalInputs.inputs,logicalInputs,tensor,hz,hh,hn,hd,
      Relocation.parameters,Relocation.offsets,ChanTy.read] using hb
  · intro z hz h hh n hn d hd
    have hb := bound (2:Fin 5) (coordinates p (⟨z,hz⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩)) trivial
    simpa [programIO,AlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,
      offset,inputRegion,LogicalInputs.inputs,logicalInputs,tensor,hz,hh,hn,hd,
      Relocation.parameters,Relocation.offsets,ChanTy.read] using hb
  · intro n hn
    have hb := bound (3:Fin 5) ⟨n,hn⟩ trivial
    simpa [programIO,AlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,
      offset,inputRegion,LogicalInputs.inputs,logicalInputs,scale,hn,arity,queryScales,
      Relocation.parameters,Relocation.offsets,ChanTy.read] using hb
  · intro n hn
    have hb := bound (4:Fin 5) ⟨n,hn⟩ trivial
    simpa [programIO,AlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,
      offset,inputRegion,LogicalInputs.inputs,logicalInputs,scale,hn,arity,keyScales,
      Relocation.parameters,Relocation.offsets,ChanTy.read] using hb

theorem scale_positive (p : Parameters) (idx : GridIndex (GridCoordinates.grid p)) :
    0<queryScales p ∧ 0<keyScales p := by
  have h0 := (idx ⟨0,by simp [GridCoordinates.grid,Grid.rank]⟩).isLt
  have h1 := (idx ⟨1,by simp [GridCoordinates.grid,Grid.rank]⟩).isLt
  change _ < (if p.headDim=0 then 0 else (p.context+127)/128) at h0
  change _ < p.batches*p.heads at h1
  have hc : 0<p.context := by split_ifs at h0 <;> omega
  exact ⟨Nat.mul_pos (by omega) (by omega),Nat.mul_pos (by omega) (by omega)⟩

theorem windows (A : FlatAlloc) (p : Parameters) (values : Inputs p)
    (s : BlockState) (bounds : RegionBounds)
    (backed : (programIO A p).InBounds bounds s values)
    (idx : GridIndex (GridCoordinates.grid p)) :
    ViewBounds.Windows bounds (Relocation.parameters p A.flat) (Relocation.offsets A p) := by
  constructor
  · intro z hz h hh n hn d hd
    have h := backed.1 (0:Fin 5) (coordinates p (⟨z,hz⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩)) trivial
    simpa [programIO,offset,inputRegion,Relocation.parameters,Relocation.offsets] using h
  · intro z hz h hh n hn d hd
    have h := backed.1 (1:Fin 5) (coordinates p (⟨z,hz⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩)) trivial
    simpa [programIO,offset,inputRegion,Relocation.parameters,Relocation.offsets] using h
  · intro z hz h hh n hn d hd
    have h := backed.1 (2:Fin 5) (coordinates p (⟨z,hz⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩)) trivial
    simpa [programIO,offset,inputRegion,Relocation.parameters,Relocation.offsets] using h
  · intro z hz h hh n hn d hd
    have h := backed.2.1 (0:Fin 1) (coordinates p (⟨z,hz⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩)) trivial
    simpa [programIO,OutputLayout.address,Relocation.parameters,Relocation.offsets] using h
  · have positive := (scale_positive p idx).1
    have hb := backed.1 (3:Fin 5) ⟨queryScales p-1,by change _<queryScales p; omega⟩ trivial
    change queryScales p-1+A.base p.QS < bounds A.flat at hb
    change queryScales p+A.base p.QS ≤ bounds A.flat
    omega
  · have positive := (scale_positive p idx).2
    have hb := backed.1 (4:Fin 5) ⟨keyScales p-1,by change _<keyScales p; omega⟩ trivial
    change keyScales p-1+A.base p.KS < bounds A.flat at hb
    change keyScales p+A.base p.KS ≤ bounds A.flat
    omega

theorem frame (A : FlatAlloc) (p : Parameters) (values : Inputs p) (s t : BlockState)
    (outside : ∀ r off, ¬GridFootprint.window A p (r,off) → t.mem r off=s.mem r off) :
    (programIO A p).Frame s t values := by
  intro r off ho _
  apply outside
  rintro ⟨region,c,address⟩
  have h := ho (0:Fin 1) (coordinates p c) trivial
  have address' : off=(programIO A p).owin (0:Fin 1) values (s.pids 0) (s.pids 1) (s.pids 2)
      (coordinates p c) := by simpa [programIO] using address.symm
  exact h.elim (fun h=>h region) (fun h=>h address')

#axiomsClean agreement
#axiomsClean windows
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
