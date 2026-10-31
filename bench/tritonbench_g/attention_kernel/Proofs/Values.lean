import bench.tritonbench_g.attention_kernel.Proofs.Body
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 32000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def key (M : Algebra α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [p.dim,p.bn]) : α :=
  if (kptr p s row).inBounds (TileShape.indexToList _ i) [1] then
    toFloat p.storage.float ((s.mem p.K
      (base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row)).read p.storage.tile)
  else M.literal (some .fp32) p.storage.float 0

def value (M : Algebra α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [p.bn,p.dim]) : α :=
  if (vptr p s row).inBounds (TileShape.indexToList _ i) [0] then
    toFloat p.storage.float ((s.mem p.V
      (base p s p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim)).read p.storage.tile)
  else M.literal (some .fp32) p.storage.float 0

def biasDynamic (M : Algebra α) (p : Parameters) (s : State α) (row : Nat) (i : Fin p.bm) : α :=
  if s.pids 0*p.bm+i.val<p.length then
    toFloat p.storage.float ((s.mem p.Bias
      (base p s p.bias+(s.pids 0*p.bm+i.val)*p.bias.row+row/p.bn)).read p.storage.tile)
  else M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 0)

def scores (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (i : TileIndex [p.bm,p.bn]) : α :=
  if i.2.1.val<p.length-row then
    M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .add
        (M.cast (some .fp32) p.storage.float .real
          (M.cast (some .fp32) .real p.storage.float (zero M)))
        (M.cast (some .fp32) .fp32 .real
          (dot.apply (some .fp32) p.storage.float p.storage.float .fp32 {} (batch:=[])
          (fun j=>ofFloat p.storage.float (query M p s j))
          (fun j=>ofFloat p.storage.float (key M p s row j)) i)))
      (M.cast (some .fp32) p.storage.float .real
        (M.binary (some .fp32) p.storage.float .mul
          (M.binary (some .fp32) p.storage.float .add
            (biasDynamic M p s row i.1) (biasStatic M p s i))
          (M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 1.44269504))))
  else M.negInf

def maximum (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc p α) (i : Fin p.bm) : α :=
  M.binary (some .fp32) .real .max (a.maximum i)
    (M.reduceMax (some .fp32) ⟨1,by simp⟩ Bool.false (scores M dot p s row) (i,()))

def probability (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc p α) (i : TileIndex [p.bm,p.bn]) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (scores M dot p s row i) (maximum M dot p s row a i.1))

def rescale (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc p α) (i : Fin p.bm) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (a.maximum i) (maximum M dot p s row a i))

def contribution (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc p α) (i : TileIndex [p.bm,p.dim]) : α :=
  M.cast (some .fp32) .fp32 .real
    (dot.apply (some .fp32) p.storage.float p.storage.float .fp32 {} (batch:=[])
    (fun j=>ofFloat p.storage.float (M.cast (some .fp32) .real p.storage.float
      (probability M dot p s row a j)))
    (fun j=>ofFloat p.storage.float (value M p s row j)) i)

/-- Source-ordered recurrence. The numerical backend is opaque here; neither
associativity nor conversion identities are used in this execution layer. -/
def update (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc p α) : Acc p α :=
  ⟨maximum M dot p s row a,
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.normalizer i) (rescale M dot p s row a i))
     (M.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false (probability M dot p s row a) (i,())),
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.value i) (rescale M dot p s row a i.1))
     (contribution M dot p s row a i)⟩

theorem regionEta (r : RegionName) : ({name:=r.name} : RegionName)=r := rfl
@[simp] theorem lane2Eta {m n : Nat} (i : TileIndex [m,n]) : (i.1,i.2.1,PUnit.unit)=i := by
  rcases i with ⟨i,j,u⟩
  cases u
  rfl
@[simp] theorem lane1Eta {m : Nat} (i : TileIndex [m]) : (i.1,PUnit.unit)=i := by
  rcases i with ⟨i,u⟩
  cases u
  rfl

theorem bias_addr (i z h row block bm strideB strideH strideR heads : Nat) :
    i*strideR+(h/heads*strideB+(h%heads*strideH+(row/block+z*bm*strideR)))=
      row/block+(i*strideR+(z*bm*strideR+(h/heads*strideB+h%heads*strideH))) := by omega

private theorem bias_projection (ph heads batch head pm bm lane stride row bn : Nat) :
    ((ph:Int)/(heads:Int)*(batch:Int)+(ph:Int)%(heads:Int)*(head:Int)+
      ((pm:Int)*(bm:Int)+(lane:Int))*(stride:Int)+(row:Int)/(bn:Int)).toNat =
      row/bn+(lane*stride+(pm*bm*stride+(ph/heads*batch+ph%heads*head))) := by
  simp only [←Int.natCast_ediv,←Int.natCast_emod,←Int.natCast_mul,←Int.natCast_add,
    Int.toNat_natCast]
  simp [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

private theorem bias_guard (pm bm lane stop : Nat) :
    ((pm:Int)*(bm:Int)+(lane:Int)<(stop:Int)) ↔ pm*bm+lane<stop := by
  norm_cast

theorem body_context (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (hbn : 0<p.bn) (ctx : Context M p initial row a current) :
    Context M p initial (row+p.bn) (update M dot p initial row a)
      (bodyExec M dot p initial current row a hbn ctx).val := by
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  rcases Bias with ⟨biasName⟩
  cases storage
  all_goals
    have hquery:=ctx.query
    have hbias:=ctx.biasStatic
    simp only [TypedSource.Storage.float,TypedSource.Storage.tile,FloatDType.toTileDType,ofFloat,id_eq] at hquery hbias
    constructor
  all_goals try simp [bodyExec,ctx.memory,ctx.aligned,ctx.startM,ctx.rows,ctx.biasOffset,ctx.outputOffset,
    hbias,hquery,ctx.biasStatic,ctx.query,ctx.keyPointer,ctx.valuePointer,ctx.maximum,ctx.normalizer,ctx.value,
    update,maximum,rescale,probability,scores,contribution,key,value,biasDynamic,zero,
    base,kptr,vptr,TypedSource.Storage.float,TypedSource.Storage.tile,FloatDType.toTileDType,ofFloat,toFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    BlockPtr.advance,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try funext i
  all_goals try simp [update,maximum,rescale,probability,scores,contribution,key,value,biasDynamic,zero,
    base,kptr,vptr,TypedSource.Storage.float,TypedSource.Storage.tile,FloatDType.toTileDType,ofFloat,toFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try delta scores maximum rescale probability contribution key value biasDynamic zero
  all_goals try simp [base,kptr,vptr,TypedSource.Storage.float,TypedSource.Storage.tile,
    FloatDType.toTileDType,ofFloat,toFloat,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try norm_num
  all_goals try simp [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try rfl
  all_goals try rw [lane1Eta i]
  all_goals try simp only [bias_addr]
  all_goals try simp [bias_projection,bias_guard,Value,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try rfl

#specNonCircular update avoiding [run,FP.Structural.exec,bodyExec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
