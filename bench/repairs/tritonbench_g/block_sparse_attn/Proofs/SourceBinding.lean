import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Wrapper

/-! External selected-source/compiler realization. The pinned repaired Python
kernel AST digest is 3d61b6f42ebad945a3b40313608a63e37ef669ecffc9a55217eb6398d1e4f9d1.
The selected configuration is B_M=B_N=B_D=16, four warps, three stages, FP32.
These bindings describe actual selected graph fragments, not a universally
verified compiler. The source pin and eight offline profiles alone do not
produce a binding. Local maps are structural pre-evaluation maps: no successful
execution, final row word, tensor equality, or mathematical target is a field.
The public execution consumes every mapped fragment and output address. -/
namespace VeriTile.Bench.Repairs.BlockSparseAttn.SourceBinding
open Physical Wrapper
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

structure StoreMap (s : Shape) where
  offset : Work → Nat
  map : offset=fun w=>w.target s

structure Binding (x : Layout) (p : Ops α) extends StoreMap x.shape where
  input : Memory α → Memory α
  input_map : input=id
  count : Nat → Nat → Nat
  count_map : count=x.csr.count x.shape
  key : Nat → Nat → Nat → Nat → Int
  key_map : key=x.csr.key x.shape
  scoreDot : Memory α → Nat → Nat → Nat → Int → α
  score_map : scoreDot=BlockSparseAttn.scoreDot x p
  value : Memory α → Nat → Nat → Int → Nat → α
  value_map : value=vLoad x p
  initial : Acc α x.reduction.replicas
  initial_map : initial=BlockSparseAttn.initial p x.reduction.replicas
  update : (Fin x.shape.blockN → α) → (Nat → α) → Nat → Acc α x.reduction.replicas → Acc α x.reduction.replicas
  update_map : update=normalizedStep x.reduction x.valueInstruction p

variable {α : Type} {x : Layout} {p : Ops α}

def block (binding : Binding x p) (m : Memory α) (scale : α) (w : Work)
    (state : Acc α x.reduction.replicas) (visit : Nat) : Acc α x.reduction.replicas :=
  binding.update
    (fun lane=>let key:=binding.key w.head w.row visit lane.val
      p.fma scale (binding.scoreDot (binding.input m) w.batch w.head w.row key)
        (if key≤(w.row:Int) then p.zero else p.negInf))
    (fun lane=>binding.value (binding.input m) w.batch w.head (binding.key w.head w.row visit lane) w.col)
    w.col state

def value (binding : Binding x p) (m : Memory α) (scale : α) (w : Work) : α :=
  ((List.range (binding.count w.head w.row)).foldl (block binding m scale w) binding.initial).output

theorem block_eq (binding : Binding x p) (m : Memory α) (scale : α) (w : Work)
    (state : Acc α x.reduction.replicas) (visit : Nat) :
    block binding m scale w state visit=blockStep x p m scale w.batch w.head w.row w.col state visit := by
  simp only [block,binding.update_map,binding.key_map,binding.score_map,binding.value_map,binding.input_map,id_eq,
    blockStep]
  rfl

theorem value_eq (binding : Binding x p) (m : Memory α) (scale : α) (w : Work) :
    value binding m scale w=rowValue x p m scale w := by
  unfold value
  rw [binding.count_map,binding.initial_map]
  simp_rw [show block binding m scale w=blockStep x p m scale w.batch w.head w.row w.col from
    funext (fun state=>funext (fun visit=>block_eq binding m scale w state visit))]
  rfl

def selectedStep (api : API β Store Byte) (zero : β) (buffers : Buffers) (s : Shape)
    (stores : StoreMap s) (compute : Memory β → Work → β) (m : Store) (q : Coord s) : Store :=
  api.write m (inputAddress buffers.output (stores.offset q.val))
    (compute (Physical.inputs api zero buffers m) q.val)

def selectedExecute (api : API β Store Byte) (zero : β) (buffers : Buffers) (s : Shape)
    (stores : StoreMap s) (compute : Memory β → Work → β) (order : List (Coord s)) (m : Store) : Store :=
  order.foldl (selectedStep api zero buffers s stores compute) m

theorem execute_eq (api : API β Store Byte) (zero : β) (buffers : Buffers) (s : Shape)
    (stores : StoreMap s) (compute : Memory β → Work → β) (order : List (Coord s)) (m : Store) :
    selectedExecute api zero buffers s stores compute order m=Physical.execute api zero buffers s compute order m := by
  unfold selectedExecute Physical.execute
  congr 1
  funext state q
  simp only [selectedStep,Physical.step,stores.map,Physical.outputAddress]

def selectedDispatch (api : API β Store Byte) (zero : β) (a : Host.Arguments) (out : Host.Tensor)
    (buffers : Buffers) (rowHeader colHeader : MetadataHeader) (rows cols : Nat)
    (stores : StoreMap a.layout.shape) (compute : Memory β → Work → β) (order : List (Coord a.layout.shape)) (m : Store) :
    Except Unit (Host.Tensor × Store) :=
  match validate a out buffers rowHeader colHeader rows cols with
  | .error _ => .error ()
  | .ok Bool.false => .ok (out,m)
  | .ok Bool.true => .ok (out,selectedExecute api zero buffers a.layout.shape stores compute order m)

theorem dispatch_eq (api : API β Store Byte) (zero : β) (a : Host.Arguments) (out : Host.Tensor)
    (buffers : Buffers) (rowHeader colHeader : MetadataHeader) (rows cols : Nat)
    (stores : StoreMap a.layout.shape) (compute : Memory β → Work → β) (order : List (Coord a.layout.shape)) (m : Store) :
    selectedDispatch api zero a out buffers rowHeader colHeader rows cols stores compute order m=
      dispatch api zero a out buffers rowHeader colHeader rows cols compute order m := by
  unfold selectedDispatch dispatch
  cases validate a out buffers rowHeader colHeader rows cols with
  | error _ => rfl
  | ok launch => cases launch <;> simp only [execute_eq]

#axiomsClean value_eq
#axiomsClean execute_eq
#axiomsClean dispatch_eq
#specNonCircular Binding avoiding [value,rowValue,Physical.execute,Host.invoke,Spec.output]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.SourceBinding
