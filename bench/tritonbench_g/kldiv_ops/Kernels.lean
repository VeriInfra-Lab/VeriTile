import VeriTile.Triton.DSL
import bench.tritonbench_g.kldiv_ops.Sources.Parameters
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KldivOps.BackwardSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.KldivOps.TypedSource

def defaultTarget (target : Storage) (output : Storage) (swapped : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  match target,output,swapped with
  | .fp32, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp32, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp32, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .fp32, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .fp32, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp32, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp32, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp32, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp32, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp32, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp32, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp32, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp64, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .fp64, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .fp64, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp64, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp64, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp64, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp64, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp64, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp64, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp64, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp64, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp64, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp16, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .fp16, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .fp16, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .fp16, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .fp16, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp16, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp16, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp16, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp16, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp16, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp16, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp16, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float16, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .bf16, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .bf16, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .bf16, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .bf16, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .bf16, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .bf16, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .bf16, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .bf16, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .bf16, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .bf16, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .bf16, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .bf16, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.bfloat16, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .f8e4, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .f8e4, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .f8e4, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .f8e4, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .f8e4, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .f8e4, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .f8e4, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .f8e4, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .f8e4, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .f8e4, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .f8e4, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .f8e4, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e4nv, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .f8e5, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .f8e5, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .f8e5, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .f8e5, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .f8e5, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .f8e5, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .f8e5, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .f8e5, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .f8e5, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .f8e5, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .f8e5, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .f8e5, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float8e5, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = target * -1
      tl.store(O + offsets, res, mask=mask)
    }
  }

def logarithmic (target : MathStorage) (output : Storage) (swapped : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  match target,output,swapped with
  | .fp32, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp32, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp32, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .fp32, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float64), mask=mask)
    }
  }
  | .fp32, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp32, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp32, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp32, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp32, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp32, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp32, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp32, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float32, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp64, .fp32, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .fp64, .fp32, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float32
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float32), mask=mask)
    }
  }
  | .fp64, .fp64, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp64, .fp64, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float64
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res, mask=mask)
    }
  }
  | .fp64, .fp16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp64, .fp16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float16), mask=mask)
    }
  }
  | .fp64, .bf16, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp64, .bf16, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.bfloat16
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.bfloat16), mask=mask)
    }
  }
  | .fp64, .f8e4, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp64, .f8e4, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e4nv
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e4nv), mask=mask)
    }
  }
  | .fp64, .f8e5, Bool.false => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += pid * $(so)
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
  | .fp64, .f8e5, Bool.true => triton {
    tl.ports inputs=[T], outputs=[O]
    tl.region T = tl.float64, O = tl.float8e5
    pid = tl.program_id(0).to(tl.int64)
    T += pid * $(st)
    O += $(so) * pid
    offsets = tl.arange(0, $(B))
    mask = offsets < $((N : Int))
    for i in range(0, $((N : Int)), $(B)) {
      offsets = i + tl.arange(0, $(B))
      mask = offsets < $((N : Int))
      target = tl.load(T + offsets, mask=mask, other=0.0)
      res = -tl.exp(target)
      tl.store(O + offsets, res.to(tl.float8e5), mask=mask)
    }
  }
end VeriTile.Bench.Repairs.KldivOps.BackwardSource

namespace VeriTile.Bench.Repairs.KldivOps
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

structure Parameters where
  (Y T O : RegionName)
  (rows cols block : Nat)
  deriving Repr

/-! Complete selected FP32 sources. Epsilon is supplied through the invocation's
runtime scalar ABI. The existing candidate commutes only the signed output-row
product, preserving every loop, branch, mask, reduction and floating operation. -/
namespace KernelDSL

def forwardDefault (p : Parameters) (red : Nat) : ComputeKernel :=
  let Y := p.Y
  let T := p.T
  let O := p.O
  triton {
    tl.ports inputs=[Y,T], outputs=[O]
    tl.region Y = tl.float32, T = tl.float32, O = tl.float32
    tl.param eps : tl.float32
    pid = tl.program_id(0).to(tl.int64)
    Y += pid * $(p.cols)
    T += pid * $(p.cols)
    O += pid * $((if red=0 then p.cols else 1 : Nat))
    base_offsets = tl.arange(0, $(p.block)).to(tl.int64)
    loss_sum = 0.0
    for i in range(0, $((p.cols : Int)), $(p.block)) {
      offsets = i + base_offsets
      mask = offsets < $((p.cols : Int))
      y = tl.load(Y + offsets, mask=mask, other=0.0)
      y_true = tl.load(T + offsets, mask=mask, other=0.0)
      loss = y_true * (tl.log(tl.maximum(y_true,eps.to(tl.float32))) - y)
      if $(red) == $(0) {
        tl.store(O + offsets, loss, mask=mask)
      } else {
        loss = tl.where(mask, loss, 0.0)
        loss_sum += tl.sum(loss, axis=0)
      }
    }
    if $(red) != $(0) { tl.store(O, loss_sum) }
  }

def forwardLog (p : Parameters) (red : Nat) : ComputeKernel :=
  let Y := p.Y
  let T := p.T
  let O := p.O
  triton {
    tl.ports inputs=[Y,T], outputs=[O]
    tl.region Y = tl.float32, T = tl.float32, O = tl.float32
    tl.param eps : tl.float32
    pid = tl.program_id(0).to(tl.int64)
    Y += pid * $(p.cols)
    T += pid * $(p.cols)
    O += pid * $((if red=0 then p.cols else 1 : Nat))
    base_offsets = tl.arange(0, $(p.block)).to(tl.int64)
    loss_sum = 0.0
    for i in range(0, $((p.cols : Int)), $(p.block)) {
      offsets = i + base_offsets
      mask = offsets < $((p.cols : Int))
      y = tl.load(Y + offsets, mask=mask, other=0.0)
      y_true = tl.load(T + offsets, mask=mask, other=0.0)
      loss = tl.exp(y_true) * (y_true - y)
      if $(red) == $(0) {
        tl.store(O + offsets, loss, mask=mask)
      } else {
        loss = tl.where(mask, loss, 0.0)
        loss_sum += tl.sum(loss, axis=0)
      }
    }
    if $(red) != $(0) { tl.store(O, loss_sum) }
  }

def forwardDefaultCommuted (p : Parameters) (red : Nat) : ComputeKernel :=
  let Y := p.Y
  let T := p.T
  let O := p.O
  triton {
    tl.ports inputs=[Y,T], outputs=[O]
    tl.region Y = tl.float32, T = tl.float32, O = tl.float32
    tl.param eps : tl.float32
    pid = tl.program_id(0).to(tl.int64)
    Y += pid * $(p.cols)
    T += pid * $(p.cols)
    O += $((if red=0 then p.cols else 1 : Nat)) * pid
    base_offsets = tl.arange(0, $(p.block)).to(tl.int64)
    loss_sum = 0.0
    for i in range(0, $((p.cols : Int)), $(p.block)) {
      offsets = i + base_offsets
      mask = offsets < $((p.cols : Int))
      y = tl.load(Y + offsets, mask=mask, other=0.0)
      y_true = tl.load(T + offsets, mask=mask, other=0.0)
      loss = y_true * (tl.log(tl.maximum(y_true,eps.to(tl.float32))) - y)
      if $(red) == $(0) {
        tl.store(O + offsets, loss, mask=mask)
      } else {
        loss = tl.where(mask, loss, 0.0)
        loss_sum += tl.sum(loss, axis=0)
      }
    }
    if $(red) != $(0) { tl.store(O, loss_sum) }
  }

def forwardLogCommuted (p : Parameters) (red : Nat) : ComputeKernel :=
  let Y := p.Y
  let T := p.T
  let O := p.O
  triton {
    tl.ports inputs=[Y,T], outputs=[O]
    tl.region Y = tl.float32, T = tl.float32, O = tl.float32
    tl.param eps : tl.float32
    pid = tl.program_id(0).to(tl.int64)
    Y += pid * $(p.cols)
    T += pid * $(p.cols)
    O += $((if red=0 then p.cols else 1 : Nat)) * pid
    base_offsets = tl.arange(0, $(p.block)).to(tl.int64)
    loss_sum = 0.0
    for i in range(0, $((p.cols : Int)), $(p.block)) {
      offsets = i + base_offsets
      mask = offsets < $((p.cols : Int))
      y = tl.load(Y + offsets, mask=mask, other=0.0)
      y_true = tl.load(T + offsets, mask=mask, other=0.0)
      loss = tl.exp(y_true) * (y_true - y)
      if $(red) == $(0) {
        tl.store(O + offsets, loss, mask=mask)
      } else {
        loss = tl.where(mask, loss, 0.0)
        loss_sum += tl.sum(loss, axis=0)
      }
    }
    if $(red) != $(0) { tl.store(O, loss_sum) }
  }

def backwardDefault (p : Parameters) : ComputeKernel :=
  BackwardSource.defaultTarget .fp32 .fp32 Bool.false p.T p.O p.cols p.cols p.cols p.block

def backwardLog (p : Parameters) : ComputeKernel :=
  BackwardSource.logarithmic .fp32 .fp32 Bool.false p.T p.O p.cols p.cols p.cols p.block

def backwardDefaultCommuted (p : Parameters) : ComputeKernel :=
  BackwardSource.defaultTarget .fp32 .fp32 Bool.true p.T p.O p.cols p.cols p.cols p.block

def backwardLogCommuted (p : Parameters) : ComputeKernel :=
  BackwardSource.logarithmic .fp32 .fp32 Bool.true p.T p.O p.cols p.cols p.cols p.block

/- Each complete quotation declares its original ABI before pointer rebinding. -/
def forward (p : Parameters) (flag : Bool) (red : Nat) : ComputeKernel :=
  if flag then forwardLog p red else forwardDefault p red

def forwardCandidate (p : Parameters) (flag : Bool) (red : Nat) : ComputeKernel :=
  if flag then forwardLogCommuted p red else forwardDefaultCommuted p red

def backward (p : Parameters) (flag : Bool) : ComputeKernel :=
  if flag then backwardLog p else backwardDefault p

def backwardCandidate (p : Parameters) (flag : Bool) : ComputeKernel :=
  if flag then backwardLogCommuted p else backwardDefaultCommuted p

end KernelDSL

/-- Compatibility names used by the existing execution and mathematical proofs. -/
abbrev forward := KernelDSL.forward
abbrev backward := KernelDSL.backward

namespace KernelDSL
 theorem forward_eq (p : Parameters) (flag : Bool) (red : Nat) :
    forward p flag red = KldivOps.forward p flag red := rfl
 theorem backward_eq (p : Parameters) (flag : Bool) :
    backward p flag = KldivOps.backward p flag := rfl
end KernelDSL
end VeriTile.Bench.Repairs.KldivOps
