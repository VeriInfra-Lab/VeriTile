import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Contract
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.NativeProgram
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Host
open Bytes

/-- Source-order guarded host body after ordinary successful empty_like.
The input data pointer is used directly; there is no copy or resolve_neg call. -/
def body (ops : Graph.Operations Word) (x y : TensorMetadata)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap) : Heap :=
  if rows x=0 ∨ columns x=0 then before
  else NativeProgram.run ops compiler (Config.tile_positive _) before
    x.dataPtr.toNat y.dataPtr.toNat (rows x) (columns x)

def invoke (ops : Graph.Operations Word) (x y : TensorMetadata)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap) :
    Except InputError (TensorMetadata × Heap) :=
  match check x with
  | .error e => .error e
  | .ok () => .ok (y,body ops x y compiler before)

theorem returns (ops : Graph.Operations Word) (x y : TensorMetadata)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap) (h : InputContract x) :
    invoke ops x y compiler before=.ok (y,body ops x y compiler before) := by
  simp [invoke,(check_iff x).mpr h]

theorem output (ops : Graph.Operations Word) (x y : TensorMetadata)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap)
    (row : Fin (rows x)) (column : Fin (columns x)) :
    loadWord (body ops x y compiler before) (y.dataPtr.toNat+4*(row.val*columns x+column.val))=
      Graph.eval ops (NativeProgram.rowInput before x.dataPtr.toNat (columns x) row)
        (RowGraph.output compiler.profile (columns x) column
          ⟨column.val%Config.tile (columns x),Nat.mod_lt _ (Config.tile_positive _)⟩) := by
  have live : ¬(rows x=0 ∨ columns x=0) := by have r:=row.isLt;have c:=column.isLt;omega
  rw [body,if_neg live]
  exact NativeProgram.stored_output ops compiler (Config.tile_positive _) before _ _ _ _ (by omega) row column

theorem byte (ops : Graph.Operations Word) (x y : TensorMetadata)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap)
    (row : Fin (rows x)) (column : Fin (columns x)) (byte : Fin 4) :
    body ops x y compiler before (y.dataPtr.toNat+4*(row.val*columns x+column.val)+byte.val)=encode (
      Graph.eval ops (NativeProgram.rowInput before x.dataPtr.toNat (columns x) row)
        (RowGraph.output compiler.profile (columns x) column
          ⟨column.val%Config.tile (columns x),Nat.mod_lt _ (Config.tile_positive _)⟩)) byte := by
  have live : ¬(rows x=0 ∨ columns x=0) := by have r:=row.isLt;have c:=column.isLt;omega
  rw [body,if_neg live]
  exact NativeProgram.stored_byte ops compiler (Config.tile_positive _) before _ _ _ _ (by omega) row column byte

theorem frame (ops : Graph.Operations Word) (x y : TensorMetadata)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap)
    (h : InputContract x) (allocated : Allocated x y) (address : Nat)
    (outside : address<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤address) :
    body ops x y compiler before address=before address := by
  have count := count_two x h.2.2.2.1
  have size : y.numel=x.numel := congrArg List.prod allocated.1
  unfold body
  split_ifs with live
  · rfl
  · exact NativeProgram.full_frame ops compiler (Config.tile_positive _) before _ _ _ _ (by omega) address (by omega)

#axiomsClean returns
#axiomsClean output
#axiomsClean frame
end VeriTile.Bench.Repairs.SoftmaxOptimize.Host
