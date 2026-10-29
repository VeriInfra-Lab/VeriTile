import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.CheckedFP
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.NativeGraph
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Bytes

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native
open VeriTile Triton FP.Structural Contract
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000
namespace NG
abbrev Term := VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.Term
abbrev Expr := VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.Expr
abbrev Instructions := VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.Instructions
abbrev Selection := VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.Selection
abbrev lower := VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.lower
abbrev eval := VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.eval
end NG
namespace Raw
abbrev State := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.Memory
abbrev Word := BitVec 32
abbrev Byte := BitVec 8
abbrev read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.read
abbrev unpack := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.unpack
abbrev ops := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.ops
abbrev byte_of_read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.byte_of_read
end Raw

/-- No untyped reduction is used by this source: the typed-sum machine below
expands each FP16 reduction into the explicitly selected scalar tree. -/
def scalarAlgebra : Algebra NG.Term :=
  VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.algebra
    (fun _ _ _ s ax _ _ _ => VeriTile.Bench.Repairs.RmsnormFusedLlama.Native.Plan.seeded (TileShape.axisDim s ax)) .forward

structure TensorWindow (tensor : TensorInfo) (width : Nat) where
  backing : Allocation
  window : Allocation
  backingId : backing.object=tensor.backing
  windowId : window.object=backing.object
  owner : window.owner=tensor.device.index
  backingOwner : backing.owner=window.owner
  pointer : window.baseByte=tensor.dataPtr.toNat
  extent : window.bytes=width*tensor.numel
  lower : backing.baseByte≤window.baseByte
  upper : window.baseByte+window.bytes≤backing.baseByte+backing.bytes

def indexBytes (i : TensorInfo) : Nat := if i.dtype=.other "int64" then 8 else 4

/-- These describe actual original storage and the optional singleton squeeze
alias. Different live allocation identities have disjoint physical spans. -/
structure Binding (x w y i : TensorInfo) where
  input : TensorWindow x 4
  weights : TensorWindow w 2
  output : TensorWindow y 4
  indices : TensorWindow i (indexBytes i)
  squeezed : TensorInfo
  squeezeShape : squeezed.shape=[rows w,width w,rank w]
  squeezePointer : squeezed.dataPtr=w.dataPtr
  squeezeBacking : squeezed.backing=w.backing
  inputSeparate : x.backing≠y.backing → input.backing.separate output.backing
  weightSeparate : w.backing≠y.backing → weights.backing.separate output.backing
  indexSeparate : i.backing≠y.backing → indices.backing.separate output.backing

def half (memory : Raw.State) (owner address : Nat) : Raw.Word :=
  (memory owner (address+1) ++ memory owner address).zeroExtend 32

def index (memory : Raw.State) (i : TensorInfo) (window : Allocation) (b : Nat) : Int :=
  let address:=window.baseByte+indexBytes i*b
  if i.dtype=.other "int64" then
    (Raw.read memory ⟨window.owner,address+4⟩ ++ Raw.read memory ⟨window.owner,address⟩).toInt
  else (Raw.read memory ⟨window.owner,address⟩).toInt

def data (a : Binding x w y i) (memory : Raw.State) : FPValues.Data NG.Term where
  input := fun off => .leaf "X" off (Raw.read memory (a.input.window.key off))
  weights := fun off => .leaf "W" off (half memory a.weights.window.owner (a.weights.window.baseByte+2*off))
  output := fun off => .leaf "Y" off (Raw.read memory (a.output.window.key off))
  indices := index memory i a.indices.window
  other := fun _ _ => .mk .real (.bits 0)
  padding := fun dtype _ _ => defaultValue dtype

def reduction (plans : FPScalarTree.Plans) := FPScalarTree.reduction scalarAlgebra plans

def reference (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (h : Call) (a : Binding x w y i) (before : Raw.State) (p : Geometry.Program h) : Raw.Word :=
  NG.eval I S (FPGrid.value scalarAlgebra (reduction plans) h p (data a before))

def productionAST : String := "Module(body=[Import(names=[alias(name='torch')]), Import(names=[alias(name='triton')]), Import(names=[alias(name='triton.language', asname='tl')]), FunctionDef(name='_bgmv_expand_slice_kernel', args=arguments(posonlyargs=[], args=[arg(arg='input_ptr'), arg(arg='lora_ptr'), arg(arg='out_ptr'), arg(arg='N'), arg(arg='K'), arg(arg='lora_indices'), arg(arg='xm_stride'), arg(arg='xk_stride'), arg(arg='l0_stride'), arg(arg='lora_k_stride'), arg(arg='lora_n_stride'), arg(arg='cm_stride'), arg(arg='cn_stride'), arg(arg='slice_offset'), arg(arg='BLOCK_N', annotation=Attribute(value=Name(id='tl', ctx=Load()), attr='constexpr', ctx=Load())), arg(arg='BLOCK_K', annotation=Attribute(value=Name(id='tl', ctx=Load()), attr='constexpr', ctx=Load())), arg(arg='SPLIT_N', annotation=Attribute(value=Name(id='tl', ctx=Load()), attr='constexpr', ctx=Load())), arg(arg='EVEN_K', annotation=Attribute(value=Name(id='tl', ctx=Load()), attr='constexpr', ctx=Load())), arg(arg='ADD_INPUTS', annotation=Attribute(value=Name(id='tl', ctx=Load()), attr='constexpr', ctx=Load())), arg(arg='CAST_TYPE', annotation=Attribute(value=Name(id='tl', ctx=Load()), attr='constexpr', ctx=Load()))], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Expr(value=Constant(value=\"\\n    GroupGEMV, introducing SPLIT_N can improve large hidden_size's performance\\n    \")), Assign(targets=[Name(id='pid_sn', ctx=Store())], value=Call(func=Attribute(value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='program_id', ctx=Load()), args=[], keywords=[keyword(arg='axis', value=Constant(value=0))]), attr='to', ctx=Load()), args=[Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[])), Assign(targets=[Name(id='cur_batch', ctx=Store())], value=Call(func=Attribute(value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='program_id', ctx=Load()), args=[], keywords=[keyword(arg='axis', value=Constant(value=1))]), attr='to', ctx=Load()), args=[Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[])), Assign(targets=[Name(id='lora_index', ctx=Store())], value=Call(func=Attribute(value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='load', ctx=Load()), args=[BinOp(left=Name(id='lora_indices', ctx=Load()), op=Add(), right=Name(id='cur_batch', ctx=Load()))], keywords=[]), attr='to', ctx=Load()), args=[Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[])), If(test=Compare(left=Name(id='lora_index', ctx=Load()), ops=[Eq()], comparators=[UnaryOp(op=USub(), operand=Constant(value=1))]), body=[Return()], orelse=[]), Assign(targets=[Name(id='offset_k', ctx=Store())], value=Call(func=Attribute(value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='arange', ctx=Load()), args=[Constant(value=0), Name(id='BLOCK_K', ctx=Load())], keywords=[]), attr='to', ctx=Load()), args=[Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[])), Assign(targets=[Name(id='offset_n', ctx=Store())], value=Call(func=Attribute(value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='arange', ctx=Load()), args=[Constant(value=0), Name(id='BLOCK_N', ctx=Load())], keywords=[]), attr='to', ctx=Load()), args=[Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[])), If(test=Name(id='EVEN_K', ctx=Load()), body=[Assign(targets=[Name(id='tiled_a', ctx=Store())], value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='load', ctx=Load()), args=[BinOp(left=BinOp(left=Name(id='input_ptr', ctx=Load()), op=Add(), right=BinOp(left=Name(id='cur_batch', ctx=Load()), op=Mult(), right=Name(id='xm_stride', ctx=Load()))), op=Add(), right=BinOp(left=Name(id='offset_k', ctx=Load()), op=Mult(), right=Name(id='xk_stride', ctx=Load())))], keywords=[]))], orelse=[Assign(targets=[Name(id='tiled_a', ctx=Store())], value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='load', ctx=Load()), args=[BinOp(left=BinOp(left=Name(id='input_ptr', ctx=Load()), op=Add(), right=BinOp(left=Name(id='cur_batch', ctx=Load()), op=Mult(), right=Name(id='xm_stride', ctx=Load()))), op=Add(), right=BinOp(left=Name(id='offset_k', ctx=Load()), op=Mult(), right=Name(id='xk_stride', ctx=Load())))], keywords=[keyword(arg='mask', value=Compare(left=Name(id='offset_k', ctx=Load()), ops=[Lt()], comparators=[Name(id='K', ctx=Load())])), keyword(arg='other', value=Constant(value=0))]))]), Assign(targets=[Name(id='split_n_length', ctx=Store())], value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='cdiv', ctx=Load()), args=[Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='cast', ctx=Load()), args=[Name(id='N', ctx=Load()), Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[]), Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='cast', ctx=Load()), args=[Name(id='SPLIT_N', ctx=Load()), Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[])], keywords=[])), If(test=Name(id='CAST_TYPE', ctx=Load()), body=[Assign(targets=[Name(id='tiled_a', ctx=Store())], value=Call(func=Attribute(value=Name(id='tiled_a', ctx=Load()), attr='to', ctx=Load()), args=[Attribute(value=Attribute(value=Name(id='lora_ptr', ctx=Load()), attr='dtype', ctx=Load()), attr='element_ty', ctx=Load())], keywords=[]))], orelse=[]), Assign(targets=[Name(id='b_ptr', ctx=Store())], value=BinOp(left=BinOp(left=Name(id='lora_ptr', ctx=Load()), op=Add(), right=BinOp(left=Name(id='l0_stride', ctx=Load()), op=Mult(), right=Name(id='lora_index', ctx=Load()))), op=Add(), right=BinOp(left=BinOp(left=Name(id='pid_sn', ctx=Load()), op=Mult(), right=Name(id='split_n_length', ctx=Load())), op=Mult(), right=Name(id='lora_k_stride', ctx=Load())))), Assign(targets=[Name(id='c_ptr', ctx=Store())], value=BinOp(left=BinOp(left=BinOp(left=Name(id='out_ptr', ctx=Load()), op=Add(), right=BinOp(left=Name(id='cur_batch', ctx=Load()), op=Mult(), right=Name(id='cm_stride', ctx=Load()))), op=Add(), right=BinOp(left=Name(id='pid_sn', ctx=Load()), op=Mult(), right=Name(id='split_n_length', ctx=Load()))), op=Add(), right=BinOp(left=Name(id='slice_offset', ctx=Load()), op=Mult(), right=Name(id='cn_stride', ctx=Load())))), For(target=Name(id='n', ctx=Store()), iter=Call(func=Name(id='range', ctx=Load()), args=[Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='cast', ctx=Load()), args=[Constant(value=0), Attribute(value=Name(id='tl', ctx=Load()), attr='int64', ctx=Load())], keywords=[]), Name(id='split_n_length', ctx=Load()), Name(id='BLOCK_N', ctx=Load())], keywords=[]), body=[Assign(targets=[Name(id='current_n', ctx=Store())], value=BinOp(left=Name(id='n', ctx=Load()), op=Add(), right=Name(id='offset_n', ctx=Load()))), Assign(targets=[Name(id='b_ptr_mask', ctx=Store())], value=BinOp(left=Compare(left=Subscript(value=Name(id='current_n', ctx=Load()), slice=Tuple(elts=[Slice(), Constant(value=None)], ctx=Load()), ctx=Load()), ops=[Lt()], comparators=[Name(id='split_n_length', ctx=Load())]), op=BitAnd(), right=Compare(left=Subscript(value=Name(id='offset_k', ctx=Load()), slice=Tuple(elts=[Constant(value=None), Slice()], ctx=Load()), ctx=Load()), ops=[Lt()], comparators=[Name(id='K', ctx=Load())]))), Assign(targets=[Name(id='c_mask', ctx=Store())], value=Compare(left=Name(id='current_n', ctx=Load()), ops=[Lt()], comparators=[Name(id='split_n_length', ctx=Load())])), Assign(targets=[Name(id='tiled_b', ctx=Store())], value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='load', ctx=Load()), args=[BinOp(left=BinOp(left=Name(id='b_ptr', ctx=Load()), op=Add(), right=BinOp(left=Subscript(value=Name(id='current_n', ctx=Load()), slice=Tuple(elts=[Slice(), Constant(value=None)], ctx=Load()), ctx=Load()), op=Mult(), right=Name(id='lora_k_stride', ctx=Load()))), op=Add(), right=BinOp(left=Subscript(value=Name(id='offset_k', ctx=Load()), slice=Tuple(elts=[Constant(value=None), Slice()], ctx=Load()), ctx=Load()), op=Mult(), right=Name(id='lora_n_stride', ctx=Load())))], keywords=[keyword(arg='mask', value=Name(id='b_ptr_mask', ctx=Load())), keyword(arg='other', value=Constant(value=0.0))])), If(test=Name(id='ADD_INPUTS', ctx=Load()), body=[Assign(targets=[Name(id='tiled_out', ctx=Store())], value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='load', ctx=Load()), args=[BinOp(left=Name(id='c_ptr', ctx=Load()), op=Add(), right=BinOp(left=Name(id='current_n', ctx=Load()), op=Mult(), right=Name(id='cn_stride', ctx=Load())))], keywords=[keyword(arg='mask', value=Name(id='c_mask', ctx=Load()))])), Assign(targets=[Name(id='accumulator', ctx=Store())], value=BinOp(left=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='sum', ctx=Load()), args=[BinOp(left=Name(id='tiled_a', ctx=Load()), op=Mult(), right=Name(id='tiled_b', ctx=Load())), Constant(value=1)], keywords=[]), op=Add(), right=Name(id='tiled_out', ctx=Load())))], orelse=[Assign(targets=[Name(id='accumulator', ctx=Store())], value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='sum', ctx=Load()), args=[BinOp(left=Name(id='tiled_a', ctx=Load()), op=Mult(), right=Name(id='tiled_b', ctx=Load())), Constant(value=1)], keywords=[]))]), Expr(value=Call(func=Attribute(value=Name(id='tl', ctx=Load()), attr='store', ctx=Load()), args=[BinOp(left=Name(id='c_ptr', ctx=Load()), op=Add(), right=BinOp(left=Name(id='current_n', ctx=Load()), op=Mult(), right=Name(id='cn_stride', ctx=Load()))), Name(id='accumulator', ctx=Load())], keywords=[keyword(arg='mask', value=Name(id='c_mask', ctx=Load()))]))], orelse=[])], decorator_list=[Attribute(value=Name(id='triton', ctx=Load()), attr='jit', ctx=Load())], type_params=[]), FunctionDef(name='_bgmv_expand_slice', args=arguments(posonlyargs=[], args=[arg(arg='inputs', annotation=Attribute(value=Name(id='torch', ctx=Load()), attr='Tensor', ctx=Load())), arg(arg='lora_b_weights', annotation=Attribute(value=Name(id='torch', ctx=Load()), attr='Tensor', ctx=Load())), arg(arg='output_tensor', annotation=Attribute(value=Name(id='torch', ctx=Load()), attr='Tensor', ctx=Load())), arg(arg='lora_indices_tensor', annotation=Attribute(value=Name(id='torch', ctx=Load()), attr='Tensor', ctx=Load())), arg(arg='slice_offset', annotation=Name(id='int', ctx=Load())), arg(arg='slice_size', annotation=Name(id='int', ctx=Load())), arg(arg='add_inputs', annotation=Name(id='bool', ctx=Load()))], kwonlyargs=[], kw_defaults=[], defaults=[Constant(value=True)]), body=[Assign(targets=[Name(id='tensors', ctx=Store())], value=Tuple(elts=[Name(id='inputs', ctx=Load()), Name(id='lora_b_weights', ctx=Load()), Name(id='output_tensor', ctx=Load()), Name(id='lora_indices_tensor', ctx=Load())], ctx=Load())), If(test=UnaryOp(op=Not(), operand=Call(func=Name(id='all', ctx=Load()), args=[GeneratorExp(elt=Call(func=Name(id='isinstance', ctx=Load()), args=[Name(id='t', ctx=Load()), Attribute(value=Name(id='torch', ctx=Load()), attr='Tensor', ctx=Load())], keywords=[]), generators=[comprehension(target=Name(id='t', ctx=Store()), iter=Name(id='tensors', ctx=Load()), ifs=[], is_async=0)])], keywords=[])), body=[Raise(exc=Call(func=Name(id='TypeError', ctx=Load()), args=[Constant(value='BGMV inputs must be tensors')], keywords=[]))], orelse=[]), If(test=Call(func=Name(id='any', ctx=Load()), args=[GeneratorExp(elt=Compare(left=Attribute(value=Name(id='t', ctx=Load()), attr='layout', ctx=Load()), ops=[NotEq()], comparators=[Attribute(value=Name(id='torch', ctx=Load()), attr='strided', ctx=Load())]), generators=[comprehension(target=Name(id='t', ctx=Store()), iter=Name(id='tensors', ctx=Load()), ifs=[], is_async=0)])], keywords=[]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV requires strided tensors')], keywords=[]))], orelse=[]), If(test=BoolOp(op=Or(), values=[Compare(left=Attribute(value=Name(id='inputs', ctx=Load()), attr='dtype', ctx=Load()), ops=[NotEq()], comparators=[Attribute(value=Name(id='torch', ctx=Load()), attr='float32', ctx=Load())]), Compare(left=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='dtype', ctx=Load()), ops=[NotEq()], comparators=[Attribute(value=Name(id='torch', ctx=Load()), attr='float16', ctx=Load())]), Compare(left=Attribute(value=Name(id='output_tensor', ctx=Load()), attr='dtype', ctx=Load()), ops=[NotEq()], comparators=[Attribute(value=Name(id='torch', ctx=Load()), attr='float32', ctx=Load())]), Compare(left=Attribute(value=Name(id='lora_indices_tensor', ctx=Load()), attr='dtype', ctx=Load()), ops=[NotIn()], comparators=[Tuple(elts=[Attribute(value=Name(id='torch', ctx=Load()), attr='int32', ctx=Load()), Attribute(value=Name(id='torch', ctx=Load()), attr='int64', ctx=Load())], ctx=Load())])]), body=[Raise(exc=Call(func=Name(id='TypeError', ctx=Load()), args=[Constant(value='BGMV checked profile is FP32 input/output, FP16 weights, signed integer metadata')], keywords=[]))], orelse=[]), If(test=BoolOp(op=Or(), values=[Compare(left=Attribute(value=Name(id='inputs', ctx=Load()), attr='ndim', ctx=Load()), ops=[NotEq()], comparators=[Constant(value=2)]), Compare(left=Attribute(value=Name(id='output_tensor', ctx=Load()), attr='ndim', ctx=Load()), ops=[NotEq()], comparators=[Constant(value=2)]), Compare(left=Attribute(value=Name(id='lora_indices_tensor', ctx=Load()), attr='ndim', ctx=Load()), ops=[NotEq()], comparators=[Constant(value=1)]), Compare(left=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='ndim', ctx=Load()), ops=[NotIn()], comparators=[Tuple(elts=[Constant(value=3), Constant(value=4)], ctx=Load())])]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV expects X[B,K], W[L,N,K] or W[L,1,N,K], Y[B,C], indices[B]')], keywords=[]))], orelse=[]), If(test=BoolOp(op=And(), values=[Compare(left=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='ndim', ctx=Load()), ops=[Eq()], comparators=[Constant(value=4)]), Compare(left=Subscript(value=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='shape', ctx=Load()), slice=Constant(value=1), ctx=Load()), ops=[NotEq()], comparators=[Constant(value=1)])]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='four-dimensional weights require a singleton dimension')], keywords=[]))], orelse=[]), If(test=UnaryOp(op=Not(), operand=Call(func=Name(id='all', ctx=Load()), args=[GeneratorExp(elt=BoolOp(op=And(), values=[Attribute(value=Name(id='t', ctx=Load()), attr='is_cuda', ctx=Load()), Compare(left=Attribute(value=Name(id='t', ctx=Load()), attr='device', ctx=Load()), ops=[Eq()], comparators=[Attribute(value=Name(id='inputs', ctx=Load()), attr='device', ctx=Load())])]), generators=[comprehension(target=Name(id='t', ctx=Store()), iter=Name(id='tensors', ctx=Load()), ifs=[], is_async=0)])], keywords=[])), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV tensors must share a CUDA device')], keywords=[]))], orelse=[]), If(test=Call(func=Name(id='any', ctx=Load()), args=[GeneratorExp(elt=BoolOp(op=Or(), values=[UnaryOp(op=Not(), operand=Call(func=Attribute(value=Name(id='t', ctx=Load()), attr='is_contiguous', ctx=Load()), args=[], keywords=[])), Call(func=Attribute(value=Name(id='t', ctx=Load()), attr='is_neg', ctx=Load()), args=[], keywords=[]), Call(func=Attribute(value=Name(id='t', ctx=Load()), attr='is_conj', ctx=Load()), args=[], keywords=[])]), generators=[comprehension(target=Name(id='t', ctx=Store()), iter=Name(id='tensors', ctx=Load()), ifs=[], is_async=0)])], keywords=[]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV requires contiguous storage without lazy views')], keywords=[]))], orelse=[]), If(test=BoolOp(op=Or(), values=[Compare(left=Call(func=Name(id='type', ctx=Load()), args=[Name(id='slice_offset', ctx=Load())], keywords=[]), ops=[IsNot()], comparators=[Name(id='int', ctx=Load())]), Compare(left=Call(func=Name(id='type', ctx=Load()), args=[Name(id='slice_size', ctx=Load())], keywords=[]), ops=[IsNot()], comparators=[Name(id='int', ctx=Load())]), Compare(left=Call(func=Name(id='type', ctx=Load()), args=[Name(id='add_inputs', ctx=Load())], keywords=[]), ops=[IsNot()], comparators=[Name(id='bool', ctx=Load())])]), body=[Raise(exc=Call(func=Name(id='TypeError', ctx=Load()), args=[Constant(value='slice coordinates must be Python integers and add_inputs a bool')], keywords=[]))], orelse=[]), Assign(targets=[Name(id='B', ctx=Store())], value=Call(func=Attribute(value=Name(id='lora_indices_tensor', ctx=Load()), attr='numel', ctx=Load()), args=[], keywords=[])), Assign(targets=[Tuple(elts=[Name(id='L', ctx=Store()), Name(id='N', ctx=Store()), Name(id='K', ctx=Store())], ctx=Store())], value=Tuple(elts=[Subscript(value=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='shape', ctx=Load()), slice=Constant(value=0), ctx=Load()), Subscript(value=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='shape', ctx=Load()), slice=UnaryOp(op=USub(), operand=Constant(value=2)), ctx=Load()), Subscript(value=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='shape', ctx=Load()), slice=UnaryOp(op=USub(), operand=Constant(value=1)), ctx=Load())], ctx=Load())), Assign(targets=[Name(id='C', ctx=Store())], value=Subscript(value=Attribute(value=Name(id='output_tensor', ctx=Load()), attr='shape', ctx=Load()), slice=Constant(value=1), ctx=Load())), If(test=BoolOp(op=Or(), values=[Compare(left=Name(id='K', ctx=Load()), ops=[Eq()], comparators=[Constant(value=0)]), Compare(left=Subscript(value=Attribute(value=Name(id='inputs', ctx=Load()), attr='shape', ctx=Load()), slice=Constant(value=1), ctx=Load()), ops=[NotEq()], comparators=[Name(id='K', ctx=Load())]), Compare(left=Name(id='B', ctx=Load()), ops=[Gt()], comparators=[Subscript(value=Attribute(value=Name(id='inputs', ctx=Load()), attr='shape', ctx=Load()), slice=Constant(value=0), ctx=Load())]), Compare(left=Name(id='B', ctx=Load()), ops=[Gt()], comparators=[Subscript(value=Attribute(value=Name(id='output_tensor', ctx=Load()), attr='shape', ctx=Load()), slice=Constant(value=0), ctx=Load())]), Compare(left=Name(id='slice_size', ctx=Load()), ops=[NotEq()], comparators=[Name(id='N', ctx=Load())]), Compare(left=Name(id='slice_offset', ctx=Load()), ops=[Lt()], comparators=[Constant(value=0)]), Compare(left=BinOp(left=Name(id='slice_offset', ctx=Load()), op=Add(), right=Name(id='N', ctx=Load())), ops=[Gt()], comparators=[Name(id='C', ctx=Load())])]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV dimensions and output slice must match')], keywords=[]))], orelse=[]), If(test=BoolOp(op=Or(), values=[Compare(left=Name(id='B', ctx=Load()), ops=[Gt()], comparators=[Constant(value=65535)]), Compare(left=Call(func=Name(id='max', ctx=Load()), args=[Name(id='L', ctx=Load()), Name(id='N', ctx=Load()), Name(id='K', ctx=Load()), Name(id='C', ctx=Load()), BinOp(left=Name(id='N', ctx=Load()), op=Mult(), right=Name(id='K', ctx=Load()))], keywords=[]), ops=[GtE()], comparators=[BinOp(left=Constant(value=2), op=Pow(), right=Constant(value=31))])]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV grid and scalar strides must fit native launch arguments')], keywords=[]))], orelse=[]), Assign(targets=[Name(id='BLOCK_K', ctx=Store())], value=Call(func=Attribute(value=Name(id='triton', ctx=Load()), attr='next_power_of_2', ctx=Load()), args=[Name(id='K', ctx=Load())], keywords=[])), If(test=BoolOp(op=Or(), values=[Compare(left=BinOp(left=Constant(value=4), op=Mult(), right=BinOp(left=Call(func=Attribute(value=Name(id='inputs', ctx=Load()), attr='numel', ctx=Load()), args=[], keywords=[]), op=Add(), right=Name(id='BLOCK_K', ctx=Load()))), ops=[GtE()], comparators=[BinOp(left=Constant(value=2), op=Pow(), right=Constant(value=63))]), Compare(left=BinOp(left=Constant(value=2), op=Mult(), right=BinOp(left=BinOp(left=Call(func=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='numel', ctx=Load()), args=[], keywords=[]), op=Add(), right=BinOp(left=Constant(value=255), op=Mult(), right=Name(id='K', ctx=Load()))), op=Add(), right=Name(id='BLOCK_K', ctx=Load()))), ops=[GtE()], comparators=[BinOp(left=Constant(value=2), op=Pow(), right=Constant(value=63))]), Compare(left=BinOp(left=Constant(value=4), op=Mult(), right=BinOp(left=Call(func=Attribute(value=Name(id='output_tensor', ctx=Load()), attr='numel', ctx=Load()), args=[], keywords=[]), op=Add(), right=Constant(value=255))), ops=[GtE()], comparators=[BinOp(left=Constant(value=2), op=Pow(), right=Constant(value=63))])]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV formed byte offsets must fit signed 64-bit arithmetic')], keywords=[]))], orelse=[]), If(test=Call(func=Name(id='any', ctx=Load()), args=[GeneratorExp(elt=Compare(left=Attribute(value=Call(func=Attribute(value=Name(id='output_tensor', ctx=Load()), attr='untyped_storage', ctx=Load()), args=[], keywords=[]), attr='_cdata', ctx=Load()), ops=[Eq()], comparators=[Attribute(value=Call(func=Attribute(value=Name(id='t', ctx=Load()), attr='untyped_storage', ctx=Load()), args=[], keywords=[]), attr='_cdata', ctx=Load())]), generators=[comprehension(target=Name(id='t', ctx=Store()), iter=Tuple(elts=[Name(id='inputs', ctx=Load()), Name(id='lora_b_weights', ctx=Load()), Name(id='lora_indices_tensor', ctx=Load())], ctx=Load()), ifs=[], is_async=0)])], keywords=[]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='BGMV output backing must be separate from its read operands')], keywords=[]))], orelse=[]), If(test=BoolOp(op=And(), values=[Name(id='B', ctx=Load()), Name(id='N', ctx=Load()), Call(func=Name(id='any', ctx=Load()), args=[GeneratorExp(elt=BoolOp(op=And(), values=[Call(func=Attribute(value=Name(id='t', ctx=Load()), attr='numel', ctx=Load()), args=[], keywords=[]), BinOp(left=Call(func=Attribute(value=Name(id='t', ctx=Load()), attr='data_ptr', ctx=Load()), args=[], keywords=[]), op=Mod(), right=Constant(value=16))]), generators=[comprehension(target=Name(id='t', ctx=Store()), iter=Tuple(elts=[Name(id='inputs', ctx=Load()), Name(id='lora_b_weights', ctx=Load()), Name(id='output_tensor', ctx=Load())], ctx=Load()), ifs=[], is_async=0)])], keywords=[])]), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='nonempty BGMV data pointers require 16-byte alignment')], keywords=[]))], orelse=[]), If(test=BoolOp(op=Or(), values=[Compare(left=Name(id='B', ctx=Load()), ops=[Eq()], comparators=[Constant(value=0)]), Compare(left=Name(id='N', ctx=Load()), ops=[Eq()], comparators=[Constant(value=0)])]), body=[Return(value=Constant(value=None))], orelse=[]), If(test=UnaryOp(op=Not(), operand=Call(func=Name(id='bool', ctx=Load()), args=[Call(func=Attribute(value=Call(func=Attribute(value=Name(id='torch', ctx=Load()), attr='all', ctx=Load()), args=[BinOp(left=Compare(left=Name(id='lora_indices_tensor', ctx=Load()), ops=[Eq()], comparators=[UnaryOp(op=USub(), operand=Constant(value=1))]), op=BitOr(), right=BinOp(left=Compare(left=Name(id='lora_indices_tensor', ctx=Load()), ops=[GtE()], comparators=[Constant(value=0)]), op=BitAnd(), right=Compare(left=Name(id='lora_indices_tensor', ctx=Load()), ops=[Lt()], comparators=[Name(id='L', ctx=Load())])))], keywords=[]), attr='item', ctx=Load()), args=[], keywords=[])], keywords=[])), body=[Raise(exc=Call(func=Name(id='ValueError', ctx=Load()), args=[Constant(value='LoRA metadata must be -1 or an in-range adapter index')], keywords=[]))], orelse=[]), Assign(targets=[Name(id='weights', ctx=Store())], value=IfExp(test=Compare(left=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='ndim', ctx=Load()), ops=[Eq()], comparators=[Constant(value=4)]), body=Call(func=Attribute(value=Name(id='lora_b_weights', ctx=Load()), attr='squeeze', ctx=Load()), args=[Constant(value=1)], keywords=[]), orelse=Name(id='lora_b_weights', ctx=Load()))), With(items=[withitem(context_expr=Call(func=Attribute(value=Attribute(value=Name(id='torch', ctx=Load()), attr='cuda', ctx=Load()), attr='device', ctx=Load()), args=[Attribute(value=Name(id='inputs', ctx=Load()), attr='device', ctx=Load())], keywords=[]))], body=[Expr(value=Call(func=Subscript(value=Name(id='_bgmv_expand_slice_kernel', ctx=Load()), slice=Tuple(elts=[Name(id='N', ctx=Load()), Name(id='B', ctx=Load())], ctx=Load()), ctx=Load()), args=[Name(id='inputs', ctx=Load()), Name(id='weights', ctx=Load()), Name(id='output_tensor', ctx=Load()), Name(id='N', ctx=Load()), Name(id='K', ctx=Load()), Name(id='lora_indices_tensor', ctx=Load()), Name(id='K', ctx=Load()), Constant(value=1), BinOp(left=Name(id='N', ctx=Load()), op=Mult(), right=Name(id='K', ctx=Load())), Name(id='K', ctx=Load()), Constant(value=1), Name(id='C', ctx=Load()), Constant(value=1), Name(id='slice_offset', ctx=Load())], keywords=[keyword(arg='BLOCK_N', value=Constant(value=256)), keyword(arg='BLOCK_K', value=Name(id='BLOCK_K', ctx=Load())), keyword(arg='SPLIT_N', value=Name(id='N', ctx=Load())), keyword(arg='EVEN_K', value=Compare(left=BinOp(left=Name(id='K', ctx=Load()), op=Mod(), right=Name(id='BLOCK_K', ctx=Load())), ops=[Eq()], comparators=[Constant(value=0)])), keyword(arg='ADD_INPUTS', value=Name(id='add_inputs', ctx=Load())), keyword(arg='CAST_TYPE', value=Constant(value=True))]))]), Return(value=Constant(value=None))], decorator_list=[Call(func=Attribute(value=Name(id='torch', ctx=Load()), attr='inference_mode', ctx=Load()), args=[], keywords=[])], returns=Constant(value=None), type_params=[])], type_ignores=[])"
def productionSHA256 : String := "ee0fa302746b202f0d019c470d80f1e1bfc6647d75709f75f05db4daf963d553"

/-- Same typed source AST, compiler-selected scalar reduction plans, and a
primitive expression binding. No numerical complete-output equation. -/
structure CompilerBinding (plans : FPScalarTree.Plans) (S : NG.Selection) where
  sourceAST : String
  sourceIdentity : sourceAST=productionAST
  compilerPlans : FPScalarTree.Plans
  planIdentity : compilerPlans=plans
  program : NG.Term → NG.Expr
  lowered : ∀ term,program term=NG.lower S term

def outputWord (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (h : Call) (a : Binding x w y i) (before : Raw.State)
    (p : Geometry.Program h) : Raw.Word :=
  ((binding.program ((FPGrid.fold scalarAlgebra (reduction plans) h (Geometry.programs h) (data a before)).output
    (FPGrid.address h p))).eval I)

def jobs (h : Call) (indices : Nat → Int) := (Geometry.programs h).filter (fun p => decide (indices p.batch≠-1))

/-- Byte interpretation of exactly the scalar stores emitted by the direct
source. This is in-place Y storage, not a Python copyback/allocation stage. -/
def sourceStores (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (h : Call) (a : Binding x w y i) (before : Raw.State) : Raw.State :=
  stores Raw.ops (fun p => a.output.window.key (FPGrid.address h p))
    (outputWord plans S I binding h a before) (jobs h (data a before).indices) before

theorem source_read (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (h : Call) (a : Binding x w y i) (before : Raw.State)
    (p : Geometry.Program h) (active : (data a before).indices p.batch≠-1) :
    Raw.read (sourceStores plans S I binding h a before) (a.output.window.key (FPGrid.address h p))=
      outputWord plans S I binding h a before p := by
  apply readback Raw.ops _ _ _ before p
    (List.mem_filter.mpr ⟨Geometry.programs_complete h p,by simpa using active⟩)
  · intro q member equal
    have eq:=congrArg Key.address equal
    simp only [Allocation.key] at eq
    have indices : FPGrid.address h q=FPGrid.address h p := by omega
    simp only [outputWord,indices]
  · intro q member different
    apply Allocation.indices_disjoint
    intro same
    exact different (congrArg a.output.window.key same.symm)

theorem source_frame (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (h : Call) (a : Binding x w y i) (before : Raw.State)
    (owner address : Nat) (outside : ∀ p : Geometry.Program h,(data a before).indices p.batch≠-1 →
      ¬(a.output.window.key (FPGrid.address h p)).covers owner address) :
    sourceStores plans S I binding h a before owner address=before owner address := by
  apply bytes_frame Raw.ops _ _ _ before owner address
  intro p member
  exact outside p (by simpa using (List.mem_filter.mp member).2)

structure Result (y : TensorInfo) where
  tensor : TensorInfo
  same : tensor=y
  returned : Unit
  bytes : Raw.State

def invoke (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (a : Binding x w y i) (before : Raw.State) : Result y :=
  ⟨y,rfl,(),sourceStores plans S I binding (call x w y i s indices ok) a before⟩

theorem whole (plans : FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (a : Binding x w y i) (before : Raw.State)
    (metadata : ∀ b,b < i.numel → (data a before).indices b=indices b) :
    let h:=call x w y i s indices ok
    let result:=invoke plans S I binding x w y i s indices ok a before
    result.tensor=y ∧ result.returned=() ∧
    FPGrid.run scalarAlgebra (reduction plans) h (Geometry.programs h) (data a before)=
      some (FPGrid.fold scalarAlgebra (reduction plans) h (Geometry.programs h) (data a before)) ∧
    (∀ p : Geometry.Program h,indices p.batch≠-1 → ∀ byte : Fin 4,
      result.bytes a.output.window.owner ((a.output.window.key (FPGrid.address h p)).address+byte.val)=
        Raw.unpack (reference plans S I h a before p) byte) ∧
    (∀ owner address,(∀ p : Geometry.Program h,indices p.batch≠-1 →
        ¬(a.output.window.key (FPGrid.address h p)).covers owner address) → result.bytes owner address=before owner address) := by
  dsimp only
  let h:=call x w y i s indices ok
  have proof:=CheckedFP.complete x w y i s indices ok scalarAlgebra plans (data a before) metadata
  refine ⟨rfl,rfl,proof.1,?_,?_⟩
  · intro p active byte
    have hp : (data a before).indices p.batch≠-1 := by rw [metadata p.batch p.batchInside];exact active
    change sourceStores plans S I binding h a before a.output.window.owner
      ((a.output.window.key (FPGrid.address h p)).address+byte.val)=_
    calc
      _=Raw.unpack (Raw.read (sourceStores plans S I binding h a before)
        (a.output.window.key (FPGrid.address h p))) byte :=
          Raw.byte_of_read (sourceStores plans S I binding h a before) (a.output.window.key (FPGrid.address h p)) byte
      _=_ := by
        rw [source_read plans S I binding h a before p hp]
        unfold outputWord
        have value : (FPGrid.fold scalarAlgebra (reduction plans) h (Geometry.programs h) (data a before)).output
          (FPGrid.address h p)=FPGrid.value scalarAlgebra (reduction plans) h p (data a before) := proof.2.1 p active
        rw [value,binding.lowered]
        rfl
  · intro owner address outside
    apply source_frame
    intro p hp
    apply outside p
    rw [metadata p.batch p.batchInside] at hp
    exact hp

#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean whole
#specNonCircular reference avoiding [FPGrid.run,FPGrid.fold,Kernels.source,FP.Structural.exec,sourceStores]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native
