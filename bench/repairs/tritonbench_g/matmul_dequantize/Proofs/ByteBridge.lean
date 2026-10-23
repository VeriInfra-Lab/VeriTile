import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.NativeBindings
import bench.repairs.tritonbench_g.matmul_dequantize.Contract
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Physical
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.ByteBridge
open VeriTile.Triton FP.Structural Host
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

abbrev Byte := BitVec 8
abbrev Memory := RegionName → Nat → Byte
abbrev decode32 := MatmulKernel.Physical.decode

def raw32 (m : Memory) (t : Host.Tensor) (i : Nat) : BitVec 32 :=
  decode32 (fun b => m t.region (4*(t.storageOffset+i)+b.val))
def raw16 (m : Memory) (t : Host.Tensor) (i : Nat) : BitVec 16 :=
  m t.region (2*(t.storageOffset+i)+1) ++ m t.region (2*(t.storageOffset+i))
def cell (m : Memory) (t : Host.Tensor) (i : Nat) : Cell Native.Expr :=
  match t.dtype with
  | .int32 => .mk .int (raw32 m t i).toInt
  | .fp16 => .mk .fp16 (.bits16 (raw16 m t i))
  | .fp32 => .mk .fp32 (.bits32 (raw32 m t i))

def entry (a : Arguments) (m : Memory) : State Native.Expr where
  mem := fun r i =>
    if r=a.q.region then cell m a.q i
    else if r=a.z.region then cell m a.z i
    else if r=a.a.region then cell m a.a i
    else if r=a.s.region then cell m a.s i
    else if r=a.c.region then cell m a.c i
    else if r=a.d.region then cell m a.d i else .mk .fp32 (.bits32 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem packed (a : Arguments) (route : Route) (h : Admissible a route) (m : Memory) : Packed a (entry a m) := by
  have apart := h.separate
  simp only [tensors,List.pairwise_cons,List.mem_cons,List.mem_singleton,forall_eq_or_imp] at apart
  have zq : a.z.region≠a.q.region := by tauto
  constructor
  · intro i hi
    exact ⟨raw32 m a.q i,by simp [entry,cell,h.packedDtype.1]⟩
  · intro i hi
    exact ⟨raw32 m a.z i,by simp [entry,cell,h.packedDtype.2,zq]⟩

theorem input_words (a : Arguments) (route : Route) (h : Admissible a route) (m : Memory) : Native.InputWords a (entry a m) := by
  have apart := h.separate
  simp only [tensors,List.pairwise_cons,List.mem_cons,List.mem_singleton,forall_eq_or_imp] at apart
  have aq : a.a.region≠a.q.region := by tauto
  have az : a.a.region≠a.z.region := by tauto
  have sq : a.s.region≠a.q.region := by tauto
  have sz : a.s.region≠a.z.region := by tauto
  have sa : a.s.region≠a.a.region := by tauto
  intro t ht i hi
  simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with rfl|rfl
  · constructor
    · intro hd; exact ⟨raw16 m a.a i,by simp [entry,cell,hd,aq,az]⟩
    · intro hd; exact ⟨raw32 m a.a i,by simp [entry,cell,hd,aq,az]⟩
  · constructor
    · intro hd; exact ⟨raw16 m a.s i,by simp [entry,cell,hd,sq,sz,sa]⟩
    · intro hd; exact ⟨raw32 m a.s i,by simp [entry,cell,hd,sq,sz,sa]⟩

abbrev Store (g : Geometry.Layout) (width : Nat) := Grid.Index g × Fin width

def address (t : Host.Tensor) (g : Geometry.Layout) (width : Nat) (x : Store g width) :=
  width*(t.storageOffset+Grid.address g x.1)+x.2.val

theorem address_injective (t : Host.Tensor) (g : Geometry.Layout) (hg : Geometry.Valid g)
    (width : Nat) (hw : 0<width) : Function.Injective (address t g width) := by
  intro x y he
  have rem := congrArg (fun n => n%width) he
  have div := congrArg (fun n => n/width) he
  simp only [address,Nat.mul_add_mod_self_left,Nat.mod_eq_of_lt x.2.isLt,Nat.mod_eq_of_lt y.2.isLt] at rem
  have bx : x.2=y.2 := Fin.ext rem
  have ix : Grid.address g x.1=Grid.address g y.1 := by
    simp only [address,Nat.mul_add_div hw,Nat.div_eq_of_lt x.2.isLt,Nat.div_eq_of_lt y.2.isLt,Nat.add_zero] at div
    omega
  exact Prod.ext (Grid.injective g hg ix) bx

def floatRead (half : Bool) (c : Cell Native.Expr) : Native.Expr :=
  if half then c.read .fp16 else c.read .fp32

def floatCell (half : Bool) (v : Native.Expr) : Cell Native.Expr :=
  if half then .mk .fp16 v else .mk .fp32 v

theorem float_read_cell (half : Bool) (v : Native.Expr) : floatRead half (floatCell half v)=v := by
  cases half <;> simp [floatRead,floatCell,Cell.read_same]

/-- Same byte addresses as the actual typed source stores. The chosen store
is unique by source coverage; no reference value is used in serialization. -/
def serialize (t : Host.Tensor) (g : Geometry.Layout) (width : Nat) (half : Bool)
    (plan : Native.Plan) (isa : Native.WordISA) (before : Memory) (state : State Native.Expr) : Memory := fun r b =>
  if hr : r=t.region then
    if hx : ∃ x : Store g width,address t g width x=b then
      let x := Classical.choose hx
      let value := (Native.lower plan (floatRead half (state.mem t.region (Grid.address g x.1))) x.1.val.1).eval isa
      value.extractLsb' (8*x.2.val) 8
    else before r b
  else before r b

theorem serialize_read (t : Host.Tensor) (g : Geometry.Layout) (hg : Geometry.Valid g)
    (width : Nat) (hw : 0<width) (half : Bool) (plan : Native.Plan) (isa : Native.WordISA)
    (before : Memory) (state : State Native.Expr) (x : Store g width) :
    serialize t g width half plan isa before state t.region (address t g width x)=
      ((Native.lower plan (floatRead half (state.mem t.region (Grid.address g x.1))) x.1.val.1).eval isa).extractLsb' (8*x.2.val) 8 := by
  have hx : ∃ y : Store g width,address t g width y=address t g width x := ⟨x,rfl⟩
  have unique : Classical.choose hx=x := address_injective t g hg width hw (Classical.choose_spec hx)
  simp [serialize,hx,unique]

def Outside (t : Host.Tensor) (g : Geometry.Layout) (width : Nat) (r : RegionName) (b : Nat) :=
  r≠t.region ∨ ∀ x : Store g width,b≠address t g width x

theorem serialize_frame (t : Host.Tensor) (g : Geometry.Layout) (width : Nat) (half : Bool)
    (plan : Native.Plan) (isa : Native.WordISA) (before : Memory) (state : State Native.Expr)
    (r : RegionName) (b : Nat) (outside : Outside t g width r b) :
    serialize t g width half plan isa before state r b=before r b := by
  rcases outside with hr|ho
  · simp [serialize,hr]
  · have hn : ¬∃ x : Store g width,address t g width x=b := by rintro ⟨x,hx⟩; exact ho x hx.symm
    simp [serialize,hn]

/-- Only scalar or one fixed K16 instruction lane is deferred. -/
structure Numerics (actual target : Native.WordISA) : Prop where
  literal16 : ∀ x,actual.literal16 x=target.literal16 x
  literal32 : ∀ x,actual.literal32 x=target.literal32 x
  narrow : ∀ x,actual.narrow x=target.narrow x
  widen : ∀ x,actual.widen x=target.widen x
  add32 : ∀ x y,actual.add32 x y=target.add32 x y
  mul32 : ∀ x y,actual.mul32 x y=target.mul32 x y
  mul16 : ∀ x y,actual.mul16 x y=target.mul16 x y
  sub16 : ∀ x y,actual.sub16 x y=target.sub16 x y
  fma16 : ∀ x y z,actual.fma16 x y z=target.fma16 x y z
  mma16 : ∀ r c x y z,actual.mma16 r c x y z=target.mma16 r c x y z

theorem numerics_eq (actual target : Native.WordISA) (h : Numerics actual target) : actual=target := by
  cases actual; cases target
  congr
  all_goals first | exact funext h.literal16 | exact funext h.literal32 | exact funext h.narrow | exact funext h.widen |
    exact funext (fun x => funext (h.add32 x)) | exact funext (fun x => funext (h.mul32 x)) |
    exact funext (fun x => funext (h.mul16 x)) | exact funext (fun x => funext (h.sub16 x)) |
    exact funext (fun x => funext (fun y => funext (h.fma16 x y))) |
    exact funext (fun r => funext (fun c => funext (fun x => funext (fun y => funext (h.mma16 r c x y)))))

#axiomsClean packed
#axiomsClean input_words
#axiomsClean address_injective
#axiomsClean serialize_read
#axiomsClean serialize_frame
#axiomsClean numerics_eq
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.ByteBridge
