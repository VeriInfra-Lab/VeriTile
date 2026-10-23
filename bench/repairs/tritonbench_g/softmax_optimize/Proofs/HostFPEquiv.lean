import bench.repairs.tritonbench_g.softmax_optimize.Kernels
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Host
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Pointers
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Profile
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.HostFPEquiv
open VeriTile Triton Bytes

/-- Independent source-order lane recurrences and selected reduction trees,
read from the original input data pointer. This contains no source execution. -/
def target (ops : Profile.Instructions) (x : TensorMetadata)
    (profile : RowGraph.Profile (Config.tile (columns x))) (before : Heap)
    (row : Fin (rows x)) (column : Fin (columns x)) : Word :=
  Graph.eval (Profile.operations ops) (NativeProgram.rowInput before x.dataPtr.toNat (columns x) row)
    (RowGraph.output profile (columns x) column
      ⟨column.val%Config.tile (columns x),Nat.mod_lt _ (Config.tile_positive _)⟩)

specification softmax_bytes (ops : Profile.Instructions) (x y : TensorMetadata)
    (admitted : InputContract x) (allocated : Allocated x y)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap) :
    Spec.Real (∃ returned after,
      Host.invoke (Profile.operations ops) x y compiler before=.ok (returned,after) ∧
      returned=y ∧ returned.shape=x.shape ∧ returned.device=x.device ∧ returned.dtype=.floating .fp32 ∧
      (∀ row column, loadWord after (returned.dataPtr.toNat+4*(row.val*columns x+column.val))=
        target ops x compiler.profile before row column) ∧
      (∀ row column (byte : Fin 4), after (returned.dataPtr.toNat+4*(row.val*columns x+column.val)+byte.val)=
        encode (target ops x compiler.profile before row column) byte) ∧
      (∀ address, address<returned.dataPtr.toNat ∨ returned.dataPtr.toNat+4*returned.numel≤address → after address=before address) ∧
      (∀ address, x.dataPtr.toNat≤address → address<x.dataPtr.toNat+4*x.numel → after address=before address)) := by
  refine ⟨y,Host.body (Profile.operations ops) x y compiler before,
    Host.returns _ x y compiler before admitted,rfl,allocated.1,allocated.2.1,allocated.2.2.1,?_,?_,?_,?_⟩
  · exact Host.output _ x y compiler before
  · exact Host.byte _ x y compiler before
  · exact Host.frame _ x y compiler before admitted allocated
  · intro address lo hi
    apply Host.frame _ x y compiler before admitted allocated address
    rcases allocated.2.2.2.2.2.2.2 with h|h <;> omega

/-- Exact i32 PID and arange widening, i64 scalar/address arithmetic and all
active four-byte addresses are proved from the same guard and Tensor span. -/
theorem native_input (x : TensorMetadata) (admitted : InputContract x)
    (span : x.dataPtr.toNat+4*x.numel≤2^64) (row : Fin (rows x)) (column : Fin (columns x)) (byte : Fin 4) :
    Native.pid32 row.val=(row.val:Int) ∧
    Native.nativeElement (columns x) row column=(Native.element (columns x) row column:Int) ∧
    Native.nativeByte x.dataPtr.toNat (columns x) row column byte=
      x.dataPtr.toNat+4*(row.val*columns x+column.val)+byte.val ∧
    x.dataPtr.toNat+4*(row.val*columns x+column.val)+byte.val<x.dataPtr.toNat+4*x.numel := by
  have live : x.numel≠0 := by
    rw [count_two x admitted.2.2.2.1]
    have r:=row.isLt;have c:=column.isLt
    exact Nat.ne_of_gt (Nat.mul_pos (by omega) (by omega))
  have compatible := Pointers.pid_compatible x admitted live
  have bytes := Pointers.input_bytes x admitted span row column byte
  exact ⟨Native.pid32_exact compatible row,
    Native.widened_element_exact (Pointers.allocation x admitted.2.2.2.1 span) compatible row column,
    bytes.1,bytes.2.1⟩

/-- Only instruction-local numerical comparisons are conditional. The domain
follows every selected recurrence/reduction/FMA node and exp2 scaling operand. -/
theorem output_exact_if_local (ops : Profile.Instructions) (reference : Ksoftmax.Words.PrimitiveReference)
    (ports : Profile.LocalExact ops reference) (x y : TensorMetadata)
    (compiler : Lowering.Compiler (Config.tile (columns x))) (before : Heap)
    (row : Fin (rows x)) (column : Fin (columns x))
    (domain : Profile.Domain ops (NativeProgram.rowInput before x.dataPtr.toNat (columns x) row)
      (RowGraph.output compiler.profile (columns x) column
        ⟨column.val%Config.tile (columns x),Nat.mod_lt _ (Config.tile_positive _)⟩)) :
    loadWord (Host.body (Profile.operations ops) x y compiler before)
      (y.dataPtr.toNat+4*(row.val*columns x+column.val))=
      target (Ksoftmax.Words.referenceOps reference) x compiler.profile before row column :=
  (Host.output _ x y compiler before row column).trans
    (Profile.exact_if_local ops reference ports _ _ domain)

#specNonCircular target avoiding [Host.invoke,Host.body,NativeProgram.run,exec]
#specNonCircular InputContract avoiding [Host.invoke,NativeProgram.run]
#axiomsClean softmax_bytes
#axiomsClean native_input
#axiomsClean output_exact_if_local

theorem native_output (x y : TensorMetadata) (admitted : InputContract x) (allocated : Allocated x y)
    (row : Fin (rows y)) (column : Fin (columns y)) (byte : Fin 4) :
    Native.nativeByte y.dataPtr.toNat (columns y) row column byte=
      y.dataPtr.toNat+4*(row.val*columns y+column.val)+byte.val ∧
    y.dataPtr.toNat+4*(row.val*columns y+column.val)+byte.val<y.dataPtr.toNat+4*y.numel ∧
    (y.dataPtr.toNat+4*(row.val*columns y+column.val))%4=0 := by
  have hy := allocated_contract x y admitted allocated
  have live : 0<y.numel := by
    rw [count_two y hy.2.2.2.1]
    have r:=row.isLt;have c:=column.isLt
    exact Nat.mul_pos (by omega) (by omega)
  exact Pointers.input_bytes y hy (allocated.2.2.2.2.2.2.1 live).2 row column byte

#axiomsClean native_output
end VeriTile.Bench.Repairs.SoftmaxOptimize.HostFPEquiv
