import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Sources.Selected
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.BmmChunkFwd
def specialize (a b : Storage) (swapped : Bool) :=
  match a, b, swapped with
  | .fp32, .fp32, Bool.false => VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource
  | .fp32, .fp32, Bool.true => (fun out_dtype => dotFirstKernel .fp32 .fp32 out_dtype .fp32)
  | .fp32, .fp16, Bool.false => (fun out_dtype => originalKernel .fp32 .fp16 out_dtype .fp16)
  | .fp32, .fp16, Bool.true => (fun out_dtype => dotFirstKernel .fp32 .fp16 out_dtype .fp16)
  | .fp32, .bf16, Bool.false => (fun out_dtype => originalKernel .fp32 .bf16 out_dtype .bf16)
  | .fp32, .bf16, Bool.true => (fun out_dtype => dotFirstKernel .fp32 .bf16 out_dtype .bf16)
  | .fp16, .fp32, Bool.false => (fun out_dtype => originalKernel .fp16 .fp32 out_dtype .fp16)
  | .fp16, .fp32, Bool.true => (fun out_dtype => dotFirstKernel .fp16 .fp32 out_dtype .fp16)
  | .fp16, .fp16, Bool.false => (fun out_dtype => originalKernel .fp16 .fp16 out_dtype .fp16)
  | .fp16, .fp16, Bool.true => (fun out_dtype => dotFirstKernel .fp16 .fp16 out_dtype .fp16)
  | .fp16, .bf16, Bool.false => (fun out_dtype => originalKernel .fp16 .bf16 out_dtype .bf16)
  | .fp16, .bf16, Bool.true => (fun out_dtype => dotFirstKernel .fp16 .bf16 out_dtype .bf16)
  | .bf16, .fp32, Bool.false => (fun out_dtype => originalKernel .bf16 .fp32 out_dtype .bf16)
  | .bf16, .fp32, Bool.true => (fun out_dtype => dotFirstKernel .bf16 .fp32 out_dtype .bf16)
  | .bf16, .fp16, Bool.false => (fun out_dtype => originalKernel .bf16 .fp16 out_dtype .bf16)
  | .bf16, .fp16, Bool.true => (fun out_dtype => dotFirstKernel .bf16 .fp16 out_dtype .bf16)
  | .bf16, .bf16, Bool.false => (fun out_dtype => originalKernel .bf16 .bf16 out_dtype .bf16)
  | .bf16, .bf16, Bool.true => (fun out_dtype => dotFirstKernel .bf16 .bf16 out_dtype .bf16)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedSource
