import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Internal algebraic model with positive-part pointer offsets. It differs
from the original Python on inactive pointer values and is NOT a replacement
source transcription. It is used only for a planned masked-load observational
bridge. This file asserts no equality to the actual source program. -/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.PositivePointerModel
open VeriTile.Triton
def convInnerBodyFactored (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat) :
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
      (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil)) (Op.ref TileDType.ptr [BHW, 1] "Input")
            (Op.expandDim ⟨0, by simp⟩
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat IIFS)
                (Op.ref TileDType.nat [BIN] "input_feat_offset"))))
          (Op.castIntToNat
            (Op.expandDim ⟨1, by simp⟩
              (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IHS))
                (Op.ref TileDType.int [BHW] "input_height_offset")))))
        (Op.castIntToNat
          (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IWS))
              (Op.ref TileDType.int [BHW] "input_width_offset"))))),
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

def conv_casts (M K N : Nat) : List Stmt := [
 .assign .fp16 [M,K] "input_block" (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [M,K] "input_block"))),
 .assign .fp16 [K,N] "weight_block" (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [K,N] "weight_block")))]

def conv_newAcc (M K N : Nat) (tf32 : Bool) : Stmt :=
 .assign .real [M,N] "accum" (.add .real (.consSame (.consSame .nil)) (.ref .real [M,N] "accum")
  (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch:=[])
   (.ref .fp16 [M,K] "input_block") (.ref .fp16 [K,N] "weight_block") {allowTF32:=some tf32})))

def conv_newBranch (M K N : Nat) (tf32 : Bool) : Stmt :=
 .ifThenElse (.constBool Bool.true) (conv_casts M K N ++ [conv_newAcc M K N tf32])
  [.assign .real [M,N] "accum" (.add .real (.consSame (.consSame .nil)) (.ref .real [M,N] "accum")
   (.dot (batch:=[]) (.ref .real [M,K] "input_block") (.ref .real [K,N] "weight_block") {allowTF32:=some tf32}))]

def convInnerBody (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat) (tf32 : Bool) : List Stmt :=
  (convInnerBodyFactored BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW).take 9 ++ [conv_newBranch BHW BIN OF tf32]

def convPostBody (Input Weight Output : RegionName)
    (batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BHW, OF] "Output"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Output)
        (Op.add NumericDType.nat (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.add NumericDType.nat (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.add NumericDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OBS)
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset")))
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OOFS)
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "group_pid")
                      (Op.ref TileDType.nat [] "out_group_dim"))
                    (Op.ref TileDType.nat [OF] "output_feat_offset")))))
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OHS)
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_height_offset"))))
          (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OWS)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))))),
    Stmt.assign .bool [BHW, OF] "output_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset")) (Op.constNat batch_dim))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [OF] "output_feat_offset"))
              (Op.ref TileDType.nat [] "out_group_dim")))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_height_offset"))
            (Op.constNat out_height)))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))
          (Op.constNat out_width))),
    Stmt.store .real [BHW, OF] (MemAccess.ptr (Op.ref TileDType.ptr [BHW, OF] "Output"))
      (Op.ref TileDType.real [BHW, OF] "accum") (MaskOpt.mask (Op.ref TileDType.bool [BHW, OF] "output_mask")) ]

/-- **Body decomposition** (locks `convInnerBody`/`convPostBody` to the kernel). -/

def convPreBody (Input Weight : RegionName)
    (in_feat_dim out_feat_dim out_height out_width IBS IIFS WOFS groups BHW OF : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [] "batch_height_width_pid" (Op.programId 0),
    Stmt.assign .nat [] "out_feat_pid" (Op.programId 1),
    Stmt.assign .nat [] "group_pid" (Op.programId 2),
    Stmt.assign .nat [] "in_group_dim"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat in_feat_dim) (Op.constNat groups)),
    Stmt.assign .nat [] "out_group_dim"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat out_feat_dim) (Op.constNat groups)),
    Stmt.assign .nat [BHW] "batch_height_width_offset"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "batch_height_width_pid") (Op.constNat BHW))
        (Op.arange BHW)),
    Stmt.assign .nat [BHW] "batch_height_offset"
      (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_width_offset")
        (Op.constNat out_width)),
    Stmt.assign .nat [BHW] "batch_offset"
      (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_offset")
        (Op.constNat out_height)),
    Stmt.assign .nat [OF] "output_feat_offset"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "out_feat_pid") (Op.constNat OF)) (Op.arange OF)),
    Stmt.assign .nat [BHW] "output_height_offset"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_offset")
        (Op.constNat out_height)),
    Stmt.assign .nat [BHW] "output_width_offset"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_width_offset")
        (Op.constNat out_width)),
    Stmt.assign .ptr [BHW, 1] "Input"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Input)
        (Op.expandDim ⟨1, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat IBS) (Op.ref TileDType.nat [BHW] "batch_offset"))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.constNat IIFS) (Op.ref TileDType.nat [] "group_pid"))
              (Op.ref TileDType.nat [] "in_group_dim"))))),
    Stmt.assign .ptr [1, OF] "Weight"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Weight)
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat WOFS) (Op.ref TileDType.nat [OF] "output_feat_offset"))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WOFS) (Op.ref TileDType.nat [] "group_pid"))
              (Op.ref TileDType.nat [] "out_group_dim"))))),
    Stmt.assign .real [BHW, OF] "accum" (Op.full [BHW, OF] (Op.const 0)) ]

/-- `body.take 14` **is** the named prologue. -/

def surface (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF : Nat) : ComputeKernel :=
  .mk [Input,Weight] [Output] ((convPreBody Input Weight in_feat_dim out_feat_dim
    out_height out_width IBS IIFS WOFS groups BHW OF ++
    [Stmt.forRange "h" 0 KH 1 [Stmt.forRange "w" 0 KW 1
      [Stmt.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat BIN)
        (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW tf32)]]] ++
    convPostBody Input Weight Output batch_dim out_height out_width OBS OOFS OHS OWS
      (in_feat_dim / groups) BHW OF).map ComputeStmt.alg)

end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.PositivePointerModel
