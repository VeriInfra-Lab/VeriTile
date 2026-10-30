import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperOutputLayout
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperNativeArguments
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeGeometry

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000

theorem matrix_row_separated {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor)
    (M N : Nat) (hs : tensor.shape=[M,N]) (stride : Int)
    (hf : tensor.strides[0]?=some stride) (hd : InnerDense tensor) (hu : CoordinateInjective tensor) :
    NativeGeometry.RowSeparated stride.toNat M N := by
  intro row other hr ho offset ha hb
  obtain ⟨a,ha,hea⟩ := Contract.inRow_iff_col.mp ha
  obtain ⟨b,hb,heb⟩ := Contract.inRow_iff_col.mp hb
  have he := hu [row,a] [other,b] (coordinate_matrix tensor M N row a hs hr ha)
    (coordinate_matrix tensor M N other b hs ho hb)
  have hia := matrix_active_offset tensor M N stride hs (hm.strides s tensor hp).1 hf hd row a ha
  have hib := matrix_active_offset tensor M N stride hs (hm.strides s tensor hp).1 hf hd other b hb
  have hi : [row,a]=[other,b] := he (hia.trans (hea.symm.trans (heb.trans hib.symm)))
  simpa using congrArg (fun xs : List Nat => xs.headD 0) hi

theorem allocated_of_some {S Value : Type} {store : TensorStore S Descriptor Value}
    {condition : Bool} {request : Wrapper.Allocation} {output : Option Tensor} {s : S}
    (h : OptionAllocated store condition request output s) (tensor : Tensor) (he : output=some tensor) :
    Allocated store request tensor s := by
  rw [he] at h
  cases condition
  · exact False.elim h
  · exact h

theorem allocated_of_enabled {S Value : Type} {store : TensorStore S Descriptor Value}
    {condition : Bool} {request : Wrapper.Allocation} {output : Option Tensor} {s : S}
    (h : OptionAllocated store condition request output s) (hc : condition=Bool.true) :
    ∃ tensor, output=some tensor ∧ Allocated store request tensor s := by
  rw [hc] at h
  cases output with
  | none => exact False.elim h
  | some tensor => exact ⟨tensor,rfl,h⟩

/-- Row independence follows from the real allocated output coordinates.
All six native stride parameters keep their actual source values; in
particular absent optional outputs still pass zero. -/
theorem launch_geometry {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hd : OutputsDense p.outputs) (r : Regions)
    (hn : [r.y1,r.y,r.rstd,r.mean,r.residualOut,r.dropoutMask].Nodup) :
    NativeGeometry.Geometry (p.flags q) r (argumentStrides args) p.validated.M p.validated.N := by
  have hxshape := h.inputs.x.shape.trans hv.x.1
  have hyshape : p.outputs.y.shape=[p.validated.M,p.validated.N] := h.outputs.1.shape.trans hxshape
  refine ⟨hn,?_,?_,?_⟩
  · exact matrix_row_separated store hm current p.outputs.y h.outputs.1.present _ _ hyshape _
      h.arguments.y hd.y hd.y_injective
  · intro henabled
    obtain ⟨output,he,ho⟩ := allocated_of_enabled h.outputs.2.1 henabled
    have hshape : output.shape=[p.validated.M,p.validated.N] := ho.shape.trans hyshape
    have hstride : output.strides[0]?=some args.strideY1 := by
      simpa only [OptionalStrideBound,he] using h.arguments.y1
    exact matrix_row_separated store hm current output ho.present _ _ hshape _ hstride
      (hd.y1 output he) (hd.y1_injective output he)
  · intro henabled
    cases he : p.outputs.residualOut with
    | none => simp [Prepared.flags,he] at henabled
    | some output =>
      have ho := allocated_of_some h.outputs.2.2.1 output he
      have hshape : output.shape=[p.validated.M,p.validated.N] := ho.shape
      have hstride : output.strides[0]?=some args.strideResidualOut := by
        simpa only [OptionalStrideBound,he] using h.arguments.residualOut
      exact matrix_row_separated store hm current output ho.present _ _ hshape _ hstride
        (hd.residualOut output he) (hd.residualOut_injective output he)

#axiomsClean matrix_row_separated
#axiomsClean allocated_of_some
#axiomsClean allocated_of_enabled
#axiomsClean launch_geometry
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
