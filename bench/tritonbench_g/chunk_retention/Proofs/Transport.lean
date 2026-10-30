import bench.tritonbench_g.chunk_retention.Sources.Source

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

structure Ops.Hom (a : Ops α) (b : Ops β) (f : α→β) : Prop where
  zero : f a.zero=b.zero
  integer : ∀ n, f (a.integer n)=b.integer n
  cast : ∀ x, f (a.cast x)=b.cast (f x)
  add : ∀ x y, f (a.add x y)=b.add (f x) (f y)
  sub : ∀ x y, f (a.sub x y)=b.sub (f x) (f y)
  mul : ∀ x y, f (a.mul x y)=b.mul (f x) (f y)
  fma : ∀ x y z, f (a.fma x y z)=b.fma (f x) (f y) (f z)
  exp2 : ∀ x, f (a.exp2 x)=b.exp2 (f x)
  log2 : ∀ x, f (a.log2 x)=b.log2 (f x)

namespace Ops.Hom
variable {a : Ops α} {b : Ops β} {f : α→β} (hom : Ops.Hom a b f)
include hom

theorem chain_map (n : Nat) (l r : Nat→α) (seed : α) :
    f (chain a n l r seed)=chain b n (fun i => f (l i)) (fun i => f (r i)) (f seed) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [chain,hom.fma,ih]
theorem block_map (width : Nat) (l r : Nat→α) (n : Nat) (seed : α) :
    f (blockChain a width l r n seed)=
      blockChain b width (fun i => f (l i)) (fun i => f (r i)) n (f seed) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [blockChain,hom.chain_map,ih]
theorem masked_map (rows cols : Nat) (x : Tile α) (r d : Nat) :
    f (masked a rows cols x r d)=masked b rows cols (fun r d => f (x r d)) r d := by
  unfold masked
  split_ifs
  · rfl
  · exact hom.zero
theorem decay_map (c : Shape) (head : Nat) (n : Int) :
    f (decay a c head n)=decay b c head n := by
  simp only [decay,logRate,hom.exp2,hom.log2,hom.sub,hom.mul,hom.integer]
theorem causal_map (c : Shape) (h r t : Nat) :
    f (causalWeight a c h r t)=causalWeight b c h r t := by
  unfold causalWeight
  split_ifs
  · exact hom.decay_map _ _ _
  · exact hom.zero

theorem nextH_map (c : Shape) (xs : Inputs α) (h chunk : Nat) (carry : Tile α) (e j : Nat) :
    f (Body.nextH a c xs h chunk carry e j)=
      Body.nextH b c (xs.map f) h chunk (fun e j => f (carry e j)) e j := by
  simp only [Body.nextH,hom.chain_map,hom.mul,hom.cast,hom.decay_map,
    kAt,vAt,hom.masked_map,Inputs.map]
theorem nextDH_map (c : Shape) (xs : Inputs α) (scale : α) (h chunk : Nat) (carry : Tile α) (e j : Nat) :
    f (Body.nextDH a c xs scale h chunk carry e j)=
      Body.nextDH b c (xs.map f) (f scale) h chunk (fun e j => f (carry e j)) e j := by
  simp only [Body.nextDH,hom.chain_map,hom.mul,hom.cast,hom.decay_map,
    qAt,gAt,hom.masked_map,Inputs.map]
theorem carry_map (c : Shape) (xs : Inputs α) (h chunk r j : Nat) (history : Tile α) :
    f (Body.outputCarry a c xs h chunk r j history)=
      Body.outputCarry b c (xs.map f) h chunk r j (fun e j => f (history e j)) := by
  simp only [Body.outputCarry,hom.block_map,hom.mul,hom.cast,hom.zero,hom.decay_map,
    qAt,hom.masked_map,Inputs.map]
theorem score_map (c : Shape) (xs : Inputs α) (h chunk r t : Nat) :
    f (Body.score a c xs h chunk r t)=Body.score b c (xs.map f) h chunk r t := by
  simp only [Body.score,hom.block_map,hom.zero,qAt,kAt,hom.masked_map,Inputs.map]
theorem output_map (c : Shape) (xs : Inputs α) (scale : α) (h chunk : Nat) (history : Tile α) (r j : Nat) :
    f (Body.output a c xs scale h chunk history r j)=
      Body.output b c (xs.map f) (f scale) h chunk (fun e j => f (history e j)) r j := by
  simp only [Body.output,hom.cast,hom.mul,hom.chain_map,hom.score_map,hom.causal_map,
    hom.carry_map,vAt,hom.masked_map,Inputs.map]
theorem ds_map (c : Shape) (xs : Inputs α) (scale : α) (h chunk r t : Nat) :
    f (Body.ds a c xs scale h chunk r t)=Body.ds b c (xs.map f) (f scale) h chunk r t := by
  simp only [Body.ds,hom.cast,hom.mul,hom.block_map,hom.zero,hom.causal_map,
    gAt,vAt,hom.masked_map,Inputs.map]
theorem dq_map (c : Shape) (xs : Inputs α) (scale : α) (h chunk : Nat) (history : Tile α) (r e : Nat) :
    f (Body.dq a c xs scale h chunk history r e)=
      Body.dq b c (xs.map f) (f scale) h chunk (fun e j => f (history e j)) r e := by
  simp only [Body.dq,hom.cast,hom.mul,hom.chain_map,hom.ds_map,hom.block_map,hom.zero,
    hom.decay_map,kAt,gAt,hom.masked_map,Inputs.map]
theorem dk_map (c : Shape) (xs : Inputs α) (scale : α) (h chunk : Nat) (history : Tile α) (r e : Nat) :
    f (Body.dk a c xs scale h chunk history r e)=
      Body.dk b c (xs.map f) (f scale) h chunk (fun e j => f (history e j)) r e := by
  simp only [Body.dk,hom.cast,hom.mul,hom.add,hom.chain_map,hom.ds_map,hom.block_map,hom.zero,
    hom.decay_map,qAt,vAt,hom.masked_map,Inputs.map]
theorem partialScore_map (c : Shape) (xs : Inputs α) (scale : α) (block h chunk t r : Nat) :
    f (Body.partialScore a c xs scale block h chunk t r)=
      Body.partialScore b c (xs.map f) (f scale) block h chunk t r := by
  simp only [Body.partialScore,hom.mul,hom.chain_map,hom.zero,hom.causal_map,
    qAt,kAt,hom.masked_map,Inputs.map]
theorem partial_map (c : Shape) (xs : Inputs α) (scale : α) (block h chunk : Nat)
    (history : Tile α) (t j : Nat) :
    f (Body.dvPartial a c xs scale block h chunk history t j)=
      Body.dvPartial b c (xs.map f) (f scale) block h chunk (fun e j => f (history e j)) t j := by
  simp only [Body.dvPartial,hom.mul,hom.cast,hom.chain_map,hom.zero,hom.partialScore_map,
    hom.decay_map,gAt,kAt,hom.masked_map,Inputs.map]
theorem sum_map (tree : SumTree) (values : Nat→α) :
    f (tree.eval a values)=tree.eval b (fun i => f (values i)) := by
  induction tree with
  | zero => exact hom.zero
  | leaf n => rfl
  | add x y ix iy => simp only [SumTree.eval,hom.add,ix,iy]
theorem history_map (c : Shape) (xs : Inputs α) (h n : Nat) :
    (fun e j => f (Reference.history a c xs h n e j))=Reference.history b c (xs.map f) h n := by
  induction n with
  | zero => funext e j; simp only [Reference.history,hom.cast,hom.masked_map,Inputs.map]
  | succ n ih => funext e j; simp only [Reference.history,hom.nextH_map,ih]
theorem reverse_map (c : Shape) (xs : Inputs α) (scale : α) (h n : Nat) :
    (fun e j => f (Reference.reverseCarry a c xs scale h n e j))=
      Reference.reverseCarry b c (xs.map f) (f scale) h n := by
  induction n with
  | zero => funext e j; simp only [Reference.reverseCarry,hom.cast,hom.masked_map,Inputs.map]
  | succ n ih => funext e j; simp only [Reference.reverseCarry,hom.nextDH_map,ih]
theorem dhistory_map (c : Shape) (xs : Inputs α) (scale : α) (h chunk : Nat) :
    (fun e j => f (Reference.dhistory a c xs scale h chunk e j))=
      Reference.dhistory b c (xs.map f) (f scale) h chunk := hom.reverse_map _ _ _ _ _
end Ops.Hom

theorem evalHom (isa : ISA) : Ops.Hom graphOps (wordOps isa) (Expr.eval isa) := by
  constructor <;> intros <;> rfl

#axiomsClean Ops.Hom.reverse_map
#axiomsClean Ops.Hom.partial_map
#axiomsClean evalHom
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
