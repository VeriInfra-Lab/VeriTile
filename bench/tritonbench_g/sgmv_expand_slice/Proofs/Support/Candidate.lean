import bench.tritonbench_g.sgmv_expand_slice.Kernels
namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexReuse
open VeriTile.Triton
def sourceHalfSingle323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w32 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w32 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w32 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w32 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w64 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w64 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w64 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.candidateKernel dtype .half .single .w64 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceFace (starts lengths adapters : MetadataWidth) :=
  match starts,lengths,adapters with
  | .w32, .w32, .w32 => sourceHalfSingle323232
  | .w32, .w32, .w64 => sourceHalfSingle323264
  | .w32, .w64, .w32 => sourceHalfSingle326432
  | .w32, .w64, .w64 => sourceHalfSingle326464
  | .w64, .w32, .w32 => sourceHalfSingle643232
  | .w64, .w32, .w64 => sourceHalfSingle643264
  | .w64, .w64, .w32 => sourceHalfSingle646432
  | .w64, .w64, .w64 => sourceHalfSingle646464

/-- The original selected FP32/FP16/FP32 source, with only duplicate tile-index
arithmetic replaced by its already computed offset_m/offset_n registers. -/
def kernel (abi : SourceABI) (p : KernelArgs) : ComputeKernel :=
  sourceFace abi.startsWidth abi.lengthsWidth abi.adaptersWidth
    .fp32 p.input p.weight p.output p.starts p.lengths p.adapters
    p.columns p.inner p.inputRowStride p.inputInnerStride p.adapterStride
    p.weightColumnStride p.weightInnerStride p.outputRowStride p.outputColumnStride
    p.slice p.batch (p.inner % 16 == 0) abi.addInputs Bool.true

end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexReuse
