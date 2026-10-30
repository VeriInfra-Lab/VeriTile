import bench.tritonbench_g.triton_conv2d_fwd.Kernels
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.PositivePointerModel
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.MaskedPointerSimulation

/-! Actual signed-pointer algebraic body, linked definitionally to the source.
Only the shared prologue, terminal store and cast/dot branch reuse definitions
from the positive-offset proof model; the input pointer expression retains
both positive and negative terms. No equality to the positive body is asserted. -/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedPointerModel
open VeriTile.Triton
def innerFactored (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [BIN] "input_feat_offset"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "c") (Op.arange BIN)),
    Stmt.assign .int [BHW] "input_height_offset"
      (Op.add NumericDType.int Broadcast.scalarL
        (Op.sub NumericDType.int Broadcast.nil (Op.castNatToInt (Op.ref TileDType.nat [] "h"))
          (Op.constInt (Int.ofNat PH)))
        (Op.mul NumericDType.int Broadcast.scalarL (Op.constInt (Int.ofNat SH))
          (Op.castNatToInt (Op.ref TileDType.nat [BHW] "output_height_offset")))),
    Stmt.assign .int [BHW] "input_width_offset"
      (Op.add NumericDType.int Broadcast.scalarL
        (Op.sub NumericDType.int Broadcast.nil (Op.castNatToInt (Op.ref TileDType.nat [] "w"))
          (Op.constInt (Int.ofNat PW)))
        (Op.mul NumericDType.int Broadcast.scalarL (Op.constInt (Int.ofNat SW))
          (Op.castNatToInt (Op.ref TileDType.nat [BHW] "output_width_offset")))),
    Stmt.assign .ptr [BHW, BIN] "curr_input_pointer"
      (Op.ptrSub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ptrSub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.ref TileDType.ptr [BHW, 1] "Input")
          (Op.expandDim ⟨0, by simp⟩
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat IIFS)
              (Op.ref TileDType.nat [BIN] "input_feat_offset")))) (Op.castIntToNat (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IHS))
              (Op.ref TileDType.int [BHW] "input_height_offset")))))
      (Op.castIntToNat (Op.sub NumericDType.int Broadcast.scalarL (Op.constInt 0) (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IHS))
              (Op.ref TileDType.int [BHW] "input_height_offset")))))) (Op.castIntToNat (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IWS))
              (Op.ref TileDType.int [BHW] "input_width_offset")))))
      (Op.castIntToNat (Op.sub NumericDType.int Broadcast.scalarL (Op.constInt 0) (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IWS))
              (Op.ref TileDType.int [BHW] "input_width_offset")))))),
    Stmt.assign .ptr [BIN, OF] "curr_weight_pointer"
      (Op.ptrAdd Broadcast.scalarR
        (Op.ptrAdd Broadcast.scalarR
          (Op.ptrAdd (Broadcast.consL (Broadcast.consR Broadcast.nil)) (Op.ref TileDType.ptr [1, OF] "Weight")
            (Op.expandDim ⟨1, by simp⟩
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat WIFS)
                (Op.ref TileDType.nat [BIN] "input_feat_offset"))))
          (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WHS) (Op.ref TileDType.nat [] "h")))
        (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WWS) (Op.ref TileDType.nat [] "w"))),
    Stmt.assign .bool [BHW, BIN] "input_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset"))
                  (Op.constNat batch_dim))
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BIN] "input_feat_offset"))
                  (Op.ref TileDType.nat [] "in_group_dim")))
              (Op.le ComparableDType.int Broadcast.scalarL (Op.constInt (Int.ofNat 0))
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_height_offset"))))
            (Op.lt ComparableDType.int Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_height_offset"))
              (Op.castNatToInt (Op.constNat in_height))))
          (Op.le ComparableDType.int Broadcast.scalarL (Op.constInt (Int.ofNat 0))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_width_offset"))))
        (Op.lt ComparableDType.int Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_width_offset"))
          (Op.castNatToInt (Op.constNat in_width)))),
    Stmt.assign .bool [BIN, OF] "weight_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BIN] "input_feat_offset"))
          (Op.ref TileDType.nat [] "in_group_dim"))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [OF] "output_feat_offset"))
          (Op.ref TileDType.nat [] "out_group_dim"))),
    Stmt.assign .real [BHW, BIN] "input_block"
      (Op.load .real (MemAccess.ptr (Op.ref TileDType.ptr [BHW, BIN] "curr_input_pointer"))
        (MaskOpt.mask (Op.ref TileDType.bool [BHW, BIN] "input_mask"))),
    Stmt.assign .real [BIN, OF] "weight_block"
      (Op.load .real (MemAccess.ptr (Op.ref TileDType.ptr [BIN, OF] "curr_weight_pointer"))
        (MaskOpt.mask (Op.ref TileDType.bool [BIN, OF] "weight_mask"))),
    Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign .fp16 [BHW, BIN] "input_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.withPrecision .real .fp32 (Op.ref TileDType.real [BHW, BIN] "input_block"))),
        Stmt.assign .fp16 [BIN, OF] "weight_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.withPrecision .real .fp32 (Op.ref TileDType.real [BIN, OF] "weight_block")))],
    Stmt.assign .real [BHW, OF] "accum"
      (Op.add NumericDType.real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref TileDType.real [BHW, OF] "accum")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BHW, BIN] "input_block"))
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BIN, OF] "weight_block")))) ]


def inner (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width
    PH PW SH SW : Nat) (tf32 : Bool) : List Stmt :=
  (innerFactored BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW).take 9 ++
    [PositivePointerModel.conv_newBranch BHW BIN OF tf32]

theorem source_body (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF : Nat) :
    (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
      out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel.body =
    PositivePointerModel.convPreBody Input Weight in_feat_dim out_feat_dim out_height out_width
      IBS IIFS WOFS groups BHW OF ++
    [Stmt.forRange "h" 0 KH 1 [Stmt.forRange "w" 0 KW 1
      [Stmt.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat BIN)
        (inner BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW tf32)]]] ++
    PositivePointerModel.convPostBody Input Weight Output batch_dim out_height out_width
      OBS OOFS OHS OWS (in_feat_dim / groups) BHW OF := by rfl

#axiomsClean source_body
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedPointerModel
