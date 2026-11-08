import bench.tritonbench_g.matmul_dequantize_int4.IO

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word bounded_integer_cast)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- This is the existing compiled native graph partition. The source quotation
and its unfused Real interpretation are unchanged. Each copied phase is read
directly from the previously proved compiled source; only its existing half-FMA
instruction is expressed with the shared typed primitive. -/
def preludeKernel (p : Parameters) : ComputeKernel :=
  .mk [p.scale,p.zero] [] (FPCompiled.prelude p)
def prefixKernel (p : Parameters) : ComputeKernel :=
  .mk [p.input,p.weight,p.scale,p.zero] [] (FPFused.loopPrefix p)
def suffixKernel (p : Parameters) : ComputeKernel := .mk [] [] (FPFused.suffix p)
def epilogueKernel (p : Parameters) : ComputeKernel := .mk [] [p.output] (FPCompiled.epilogue p)

/-- Integer-to-half conversion is the existing exact nibble conversion. The
zero product has already rounded before sign-bit negation and the half FMA. -/
def fmaStmt (p : Parameters) : ComputeStmt :=
  .assign .fp16 [p.blockInner,p.blockColumns] "b"
    (.fmaTyped .fp16 (.leadR (.consSame .nil)) (.leadR (.consSame .nil))
      (.alg (.withPrecision .fp16 .fp32 (.castFloat .real .fp16
        (.intToReal (.ref .int [p.blockInner,p.blockColumns] "b")))))
      (.alg (.ref .fp16 [p.blockColumns] "scales"))
      (.alg (.negFloat .fp16 (.ref .fp16 [p.blockColumns] "zeros"))))

def fmaKernel (p : Parameters) : ComputeKernel := .mk [] [] [fmaStmt p]

def fmaModel (actual : Native.Instructions) : Algebra Word :=
  let base := FPSoftware.scalar FPCompiled.unused
  { base with
    unary := fun precision op x=>match op with
      | .neg .fp16=>QuantizeKvTransform.MixedBits.embed (QuantizeKvTransform.MixedBits.half x).neg
      | _=>base.unary precision op x
    fmaTyped := fun dtype=>if dtype=.fp16 then some (fun a b c=>
      QuantizeKvTransform.MixedBits.embed (actual.fma
        (QuantizeKvTransform.MixedBits.half a) (QuantizeKvTransform.MixedBits.half b)
        (QuantizeKvTransform.MixedBits.half c))) else none }

def model (actual : Native.Instructions) (op : Opcode) : Algebra Word :=
  FPCompiled.backend (Native.scalar actual) (ISA.dot (actual.mma op))

/-- A graph phase shares its predecessor registers and memory. -/
def held (kernel : ComputeKernel) (backend : Algebra Word)
    (arguments : List (RegisterBinding Word) := []) : Invocation Word where
  kernel := kernel
  backend := backend
  arguments := arguments
  pids := fun _=>0
  setPids := fun _=>Bool.false

def counter (k : Nat) : RegisterBinding Word := ⟨"k",.nat,[],fun _=>k⟩

def blockJobs (actual : Native.Instructions) (op : Opcode) (p : Parameters) (k : Nat) :
    List (Invocation Word) :=
  [held (prefixKernel p) (model actual op) [counter k],
   held (fmaKernel p) (fmaModel actual),
   held (suffixKernel p) (model actual op)]

def loopJobs (actual : Native.Instructions) (op : Opcode) (p : Parameters)
    (k : Nat) : Nat → List (Invocation Word)
  | 0=>[]
  | count+1=>blockJobs actual op p k++loopJobs actual op p (k+1) count

def programJobs (actual : Native.Instructions) (op : Opcode) (p : Parameters) (pid : Nat) :
    List (Invocation Word) :=
  ([IOExecution.launch (preludeKernel p) (model actual op) pid] ++
    loopJobs actual op p 0 (steps p)) ++ [held (epilogueKernel p) (model actual op)]

def jobs (actual : Native.Instructions) (c : RawKernel.Layout) (i : Fin 17) :
    List (Invocation Word) :=
  (List.range (KernelSource.count c.call i)).flatMap
    (programJobs actual (tunings i).opcode (params c.call i))

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeGraph
