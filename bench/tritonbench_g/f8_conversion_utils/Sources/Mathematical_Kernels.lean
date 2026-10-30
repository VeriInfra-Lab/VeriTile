import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Complete fp8 conversion kernels, retaining typed pointers, omitted load
padding and implicit destination conversion. The two identical stores in the
toHalf source are both retained. The candidate commutes only the integer
block index. Host reinterpret/launch operations are outside these kernels. -/
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
open VeriTile.Triton
inductive Format where
  | e4 | e5
  deriving DecidableEq
inductive Input where
  | fp16 | fp32
  deriving DecidableEq

abbrev sourceType := RegionName → RegionName → Nat → Nat → ComputeKernel

def toHalf (format : Format) (swapped : Bool) : sourceType :=
  match format,swapped with
  | .e4,Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float8e4nv, Y = tl.float16
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
}
  | .e4,Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float8e4nv, Y = tl.float16
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
}
  | .e5,Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float8e5, Y = tl.float16
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
}
  | .e5,Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float8e5, Y = tl.float16
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
  tl.store(Y + offs, x.to(tl.float16), mask=mask)
}

def toFP8 (input : Input) (format : Format) (swapped : Bool) : sourceType :=
  match input,format,swapped with
  | .fp16,.e4,Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float16, Y = tl.float8e4nv
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e4nv), mask=mask)
}
  | .fp16,.e4,Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float16, Y = tl.float8e4nv
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e4nv), mask=mask)
}
  | .fp16,.e5,Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float16, Y = tl.float8e5
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e5), mask=mask)
}
  | .fp16,.e5,Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float16, Y = tl.float8e5
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e5), mask=mask)
}
  | .fp32,.e4,Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float32, Y = tl.float8e4nv
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e4nv), mask=mask)
}
  | .fp32,.e4,Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float32, Y = tl.float8e4nv
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e4nv), mask=mask)
}
  | .fp32,.e5,Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float32, Y = tl.float8e5
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e5), mask=mask)
}
  | .fp32,.e5,Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float32, Y = tl.float8e5
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float8e5), mask=mask)
}

def realForward (swapped : Bool) : sourceType :=
  match swapped with
  | Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float64, Y = tl.float64
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float64), mask=mask)
  tl.store(Y + offs, x.to(tl.float64), mask=mask)
}
  | Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float64, Y = tl.float64
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float64), mask=mask)
  tl.store(Y + offs, x.to(tl.float64), mask=mask)
}

def realReverse (swapped : Bool) : sourceType :=
  match swapped with
  | Bool.false => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float64, Y = tl.float64
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float64), mask=mask)
}
  | Bool.true => fun (Y X : RegionName) (N BLOCK_SIZE : Nat) => triton {
  tl.region X = tl.float64, Y = tl.float64
  pid = tl.program_id(0)
  offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to(tl.float64), mask=mask)
}

/- Compatibility spelling only: each case delegates to a complete ordinary
quotation above. These aliases generate no kernel statements or AST nodes. -/
syntax "f8Forward%" "[" tritonDType "," tritonDType "," ident "]" : term
syntax "f8Reverse%" "[" tritonDType "," tritonDType "," ident "]" : term
open Lean
macro_rules
  | `(f8Forward%[tl.float8e4nv,tl.float16,original]) => `(toHalf .e4 Bool.false)
  | `(f8Forward%[tl.float8e5,tl.float16,original]) => `(toHalf .e5 Bool.false)
  | `(f8Forward%[tl.float64,tl.float64,original]) => `(realForward Bool.false)
  | `(f8Forward%[tl.float8e4nv,tl.float16,indexCommuted]) => `(toHalf .e4 Bool.true)
  | `(f8Forward%[tl.float8e5,tl.float16,indexCommuted]) => `(toHalf .e5 Bool.true)
  | `(f8Forward%[tl.float64,tl.float64,indexCommuted]) => `(realForward Bool.true)
  | `(f8Reverse%[tl.float16,tl.float8e4nv,original]) => `(toFP8 .fp16 .e4 Bool.false)
  | `(f8Reverse%[tl.float16,tl.float8e5,original]) => `(toFP8 .fp16 .e5 Bool.false)
  | `(f8Reverse%[tl.float32,tl.float8e4nv,original]) => `(toFP8 .fp32 .e4 Bool.false)
  | `(f8Reverse%[tl.float32,tl.float8e5,original]) => `(toFP8 .fp32 .e5 Bool.false)
  | `(f8Reverse%[tl.float64,tl.float64,original]) => `(realReverse Bool.false)
  | `(f8Reverse%[tl.float16,tl.float8e4nv,indexCommuted]) => `(toFP8 .fp16 .e4 Bool.true)
  | `(f8Reverse%[tl.float16,tl.float8e5,indexCommuted]) => `(toFP8 .fp16 .e5 Bool.true)
  | `(f8Reverse%[tl.float32,tl.float8e4nv,indexCommuted]) => `(toFP8 .fp32 .e4 Bool.true)
  | `(f8Reverse%[tl.float32,tl.float8e5,indexCommuted]) => `(toFP8 .fp32 .e5 Bool.true)
  | `(f8Reverse%[tl.float64,tl.float64,indexCommuted]) => `(realReverse Bool.true)
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
