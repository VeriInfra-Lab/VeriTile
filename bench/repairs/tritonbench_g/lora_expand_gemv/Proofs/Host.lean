import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Host
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def input (g : Geometry) : TritonBenchG.LoraExpandGemv.Host.Tensor := ⟨"X",.single,[(g.B,g.xm),(g.K,g.xk)]⟩
def output (g : Geometry) : TritonBenchG.LoraExpandGemv.Host.Tensor := ⟨"Y",.single,[(g.B,g.ym),(g.N,g.yn)]⟩
def weight (g : Geometry) : TritonBenchG.LoraExpandGemv.Host.Tensor := ⟨"W",.half,[(g.L,g.l0),(g.N,g.wn),(g.K,g.wk)]⟩
def weightArgument (g : Geometry) (four : Bool) (singletonStride : Nat) : TritonBenchG.LoraExpandGemv.Host.Tensor :=
  if four then ⟨"W",.half,[(g.L,g.l0),(1,singletonStride),(g.N,g.wn),(g.K,g.wk)]⟩ else weight g

def arguments (g : Geometry) (four : Bool) (singletonStride : Nat) : TritonBenchG.LoraExpandGemv.Host.Arguments where
  inputs := input g
  weights := weightArgument g four singletonStride
  output := output g
  indices := ⟨"I",.other "int32",[(g.B,1)]⟩
  addInputs := g.add

def launch (g : Geometry) : TritonBenchG.LoraExpandGemv.Host.Launch where
  inputType := .single
  weightType := .half
  weightView := weight g
  outputType := .single
  indexType := .other "int32"
  parameters := g.parameters
  batches := g.B

structure Contiguous (g : Geometry) : Prop where
  x : (input g).contiguous=Bool.true
  w : (weight g).contiguous=Bool.true
  y : (output g).contiguous=Bool.true

theorem host_prepare (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (c : Contiguous g) (four : Bool) (singletonStride : Nat) :
    TritonBenchG.LoraExpandGemv.Host.prepare (arguments g four singletonStride)=.ok (launch g) := by
  have hp : g.block≠0 := by
    have := TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo_pos g.K v.inner
    change 0<g.block at this
    omega
  have cx := c.x
  have cw := c.w
  have cy := c.y
  simp only [input] at cx
  simp only [weight] at cw
  simp only [output] at cy
  cases four <;>
    simp [arguments,weightArgument,input,weight,output,launch,TritonBenchG.LoraExpandGemv.Host.prepare,TritonBenchG.LoraExpandGemv.Host.checkInput,TritonBenchG.LoraExpandGemv.Host.checkWeight,
      TritonBenchG.LoraExpandGemv.Host.get,TritonBenchG.LoraExpandGemv.Host.require,TritonBenchG.LoraExpandGemv.Host.Tensor.size?,TritonBenchG.LoraExpandGemv.Host.Tensor.lastSize?,TritonBenchG.LoraExpandGemv.Host.Tensor.stride?,TritonBenchG.LoraExpandGemv.Host.Tensor.rank,
      TritonBenchG.LoraExpandGemv.Host.Tensor.squeezeOne,Geometry.parameters,Geometry.even,Geometry.block,TritonBenchG.LoraExpandGemv.Host.chooseCast,
      bind,Except.bind,pure,Except.pure,Region.cast,cx,cw,cy,show TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo g.K≠0 from hp]

/-- Tensor API binding: actual data pointers and strides, same CUDA device,
ordinary contiguous copy of logical INT32 cells, no Y/read-only overlap.
These are caller/allocator facts; no output or execution result occurs here. -/
structure TensorBinding where
  objectIdentity : Nat
  allocation : Nat
  byteBase : Nat
  cudaDevice : Nat

/-- X/W/Y bases are their data pointers. The indices base denotes backing storage;
`g.indexStart` is the original index view offset in INT32 elements. -/
structure ABI (g : Geometry) where
  x : TensorBinding
  w : TensorBinding
  y : TensorBinding
  indices : TensorBinding
  owner : Nat
  sameDevice : x.cudaDevice=owner ∧ w.cudaDevice=owner ∧ y.cudaDevice=owner ∧ indices.cudaDevice=owner
  separateOutput : y.allocation≠x.allocation ∧ y.allocation≠w.allocation ∧ y.allocation≠indices.allocation
  xAddressable : x.byteBase+4*g.xCapacity<2^64
  wAddressable : w.byteBase+2*g.wCapacity<2^64
  yAddressable : y.byteBase+4*g.yCapacity<2^64
  indexAddressable : indices.byteBase+4*g.indexCapacity<2^64

structure Preparation (g : Geometry) where
  raw : Nat → Int
  valid : Valid g raw
  contiguous : Contiguous g
  four : Bool
  singletonStride : Nat
  abi : ABI g
  callerDevice : Nat
  ownerIsCurrent : callerDevice=abi.owner

def Preparation.outputIdentity {g : Geometry} (call : Preparation g) := call.abi.y.objectIdentity

structure Result where
  memory : Memory
  returned : Option Unit
  outputIdentity : Nat
  callerDevice : Nat
  grid : Nat × Nat

def invoke (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (call : Preparation g) (schedule : List Nat) : Option Result :=
  match TritonBenchG.LoraExpandGemv.Host.prepare (arguments g call.four call.singletonStride) with
  | .error _ => none
  | .ok _ => (execute rest plans g schedule (entry g call.raw)).map fun final =>
      ⟨final,none,call.outputIdentity,call.callerDevice,(g.N,g.B)⟩

theorem whole_call (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (call : Preparation g) (schedule : List Nat) (all : schedule.Perm (List.range g.count)) :
    ∃ result,invoke rest plans g call schedule=some result ∧ result.returned=none ∧
      result.outputIdentity=call.outputIdentity ∧ result.callerDevice=call.callerDevice ∧
      result.grid=(g.N,g.B) ∧
      (∀ b n,b<g.B → n<g.N → result.memory "Y" (yoff g b n)=resultCell plans g call.raw b n) ∧
      (∀ r o,(r≠("Y" : RegionName) ∨ ∀ b n,b<g.B → n<g.N → o≠yoff g b n) →
        result.memory r o=entry g call.raw r o) := by
  obtain ⟨final,he,hv,hf⟩ := whole_source rest plans g call.raw call.valid schedule all
  refine ⟨⟨final,none,call.outputIdentity,call.callerDevice,(g.N,g.B)⟩,?_,rfl,rfl,rfl,rfl,hv,hf⟩
  simp [invoke,host_prepare g call.raw call.valid call.contiguous call.four call.singletonStride,he]

#axiomsClean host_prepare
#axiomsClean whole_call
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Host
