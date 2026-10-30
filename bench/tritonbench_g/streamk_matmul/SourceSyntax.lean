import bench.tritonbench_g.streamk_matmul.SourceReader
import VeriTile.Triton.DSL.Syntax

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Syntax
open Lean VeriTile.Triton.DSL Reader
macro "true" : term => `(Bool.true)
macro "false" : term => `(Bool.false)
syntax "while " tritonExpr " { " tritonStmt* " }" : tritonStmt
syntax "tl.full(" "(" ")" ", " tritonExpr ", " tritonDType ")" : tritonExpr
syntax ident ", " ident " = " "swizzle_tile(" tritonExpr,* ")" : tritonStmt
syntax ident ", " ident " = " "linear_tile(" tritonExpr,* ")" : tritonStmt
syntax "mac_loop(" tritonExpr,* ")" : tritonStmt
syntax "return " tritonExpr ", " tritonExpr : tritonStmt
syntax "for " ident " in " "range(0, " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax "end" : tritonExpr
syntax "end" " = " tritonExpr : tritonStmt
syntax "streamk_triton " "{" tritonStmt* "}" : term

private def name (n : TSyntax `ident) : TSyntax `term := ⟨Lean.Syntax.mkStrLit n.getId.toString⟩

private partial def expr (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match e with
  | `(tritonExpr| ($x:tritonExpr)) => expr x
  | `(tritonExpr| end) => `(Expr.variable "end")
  | `(tritonExpr| $n:num) => `(Expr.literal $n)
  | `(tritonExpr| tl.program_id(0)) => `(Expr.pid)
  | `(tritonExpr| tl.cast($x:tritonExpr, tl.int64))
  | `(tritonExpr| tl.full((), $x:tritonExpr, tl.int64)) => do
    let x ← expr x; `(Expr.signed64 $x)
  | `(tritonExpr| tl.arange(0, $n:ident)) =>
    match n.getId.toString with
    | "BLOCK_M" => `(Expr.laneM)
    | "BLOCK_N" => `(Expr.laneN)
    | "BLOCK_K" => `(Expr.laneK)
    | _ => Macro.throwErrorAt e "unsupported block range"
  | `(tritonExpr| $x:tritonExpr[:, None])
  | `(tritonExpr| $x:tritonExpr[None, :]) => expr x
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Expr.add $a $b)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Expr.sub $a $b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Expr.mul $a $b)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Expr.div $a $b)
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Expr.rem $a $b)
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) => do
    let a ← expr a; let b ← expr b; `(Expr.minimum $a $b)
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => do
    let a ← expr a; let b ← expr b; `(Expr.cdiv $a $b)
  | `(tritonExpr| $x:ident) => do let n := name x; `(Expr.variable $n)
  | _ => Macro.throwErrorAt e "unsupported stream-K integer expression"

private partial def test (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match e with
  | `(tritonExpr| ($x:tritonExpr)) => test x
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Test.eq $a $b)
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Test.ne $a $b)
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Test.lt $a $b)
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Test.gt $a $b)
  | `(tritonExpr| $a:tritonExpr & $b:tritonExpr) => do
    let a ← test a; let b ← test b; `(Test.and $a $b)
  | `(tritonExpr| $x:ident) => do let n := name x; `(Test.variable $n)
  | _ => Macro.throwErrorAt e "unsupported stream-K condition"

private partial def pointer (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match e with
  | `(tritonExpr| ($x:tritonExpr)) => pointer x
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
    let a ← pointer a; let b ← expr b; `(Pointer.offset $a $b)
  | `(tritonExpr| $x:ident) => do let n := name x; `(Pointer.variable $n)
  | _ => Macro.throwErrorAt e "unsupported stream-K pointer expression"

private def checkArgs (args : Array (TSyntax `tritonExpr)) (names : List String) : MacroM Unit := do
  unless args.size == names.length do Macro.throwError "wrong helper arity"
  for e in args, n in names do
    match e with
    | `(tritonExpr| $x:ident) =>
      unless x.getId.toString == n do Macro.throwErrorAt e "wrong helper argument"
    | _ => Macro.throwErrorAt e "helper arguments must preserve the source bindings"

private partial def stmt (s : TSyntax `tritonStmt) : MacroM (TSyntax `term) :=
  match s with
  | `(tritonStmt| end = $e:tritonExpr) => do
    let e ← expr e; `(Statement.integer "end" $e)
  | `(tritonStmt| return $a:tritonExpr, $b:tritonExpr) => do
    let a ← expr a; let b ← expr b; `(Statement.returnPair $a $b)
  | `(tritonStmt| $m:ident, $n:ident = swizzle_tile($args:tritonExpr,*)) => do
    unless m.getId.toString == "pid_m" && n.getId.toString == "pid_n" do
      Macro.throwErrorAt s "unsupported map output registers"
    let args := args.getElems
    unless args.size == 8 do Macro.throwErrorAt s "wrong helper arity"
    checkArgs (args.extract 1 8) ["M","N","K","BLOCK_M","BLOCK_N","BLOCK_K","GROUP_M"]
    let t ← expr args[0]!; `(Statement.tileCall Bool.true $t)
  | `(tritonStmt| $m:ident, $n:ident = linear_tile($args:tritonExpr,*)) => do
    unless m.getId.toString == "pid_m" && n.getId.toString == "pid_n" do
      Macro.throwErrorAt s "unsupported map output registers"
    let args := args.getElems
    unless args.size == 8 do Macro.throwErrorAt s "wrong helper arity"
    checkArgs (args.extract 1 8) ["M","N","K","BLOCK_M","BLOCK_N","BLOCK_K","GROUP_M"]
    let t ← expr args[0]!; `(Statement.tileCall Bool.false $t)
  | `(tritonStmt| mac_loop($args:tritonExpr,*)) => do
    checkArgs args.getElems ["A","B","C","M","N","K","partials","pid","stride_am","stride_ak","stride_bk","stride_bn","stride_cm","stride_cn","iters_per_tile","start_iter","end_iter","BLOCK_M","BLOCK_N","BLOCK_K","ACC_TYPE","GROUP_M"]
    `(Statement.macCall)
  | `(tritonStmt| $x:ident = tl.zeros([$m:tritonExpr, $n:tritonExpr], dtype=tl.float32)) => do
    let m ← expr m; let n ← expr n; let x := name x; `(Statement.zeros $x $m $n)
  | `(tritonStmt| $x:ident = tl.cast($y:ident, tl.float16)) => do
    unless x.getId == y.getId do Macro.throwErrorAt s "unsupported conversion target"
    let x := name x; `(Statement.half $x)
  | `(tritonStmt| $x:ident = tl.load($p:tritonExpr, mask=$m:tritonExpr, other=0.0)) => do
    let p ← pointer p; let m ← test m; let x := name x; `(Statement.load $x $p (some $m))
  | `(tritonStmt| $x:ident = tl.load($p:tritonExpr)) => do
    let p ← pointer p; let x := name x; `(Statement.load $x $p none)
  | `(tritonStmt| $x:ident += tl.dot($a:ident, $b:ident)) => do
    let x := name x; let a := name a; let b := name b; `(Statement.dotAdd $x $a $b)
  | `(tritonStmt| $x:ident += $e:tritonExpr) => do
    unless x.getId.toString ∈ ["A","B"] do Macro.throwErrorAt s "unsupported inplace update"
    let e ← expr e; let x := name x; `(Statement.pointer $x (Pointer.offset (Pointer.variable $x) $e))
  | `(tritonStmt| $x:ident = $e:tritonExpr) => do
    let n := name x
    if x.getId.toString ∈ ["A","B","C","C_"] then
      let e ← pointer e; `(Statement.pointer $n $e)
    else if x.getId.toString == "mask" then
      let e ← test e; `(Statement.boolean $n $e)
    else
      let e ← expr e; `(Statement.integer $n $e)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:ident, mask=$m:tritonExpr)) => do
    let p ← pointer p; let m ← test m; let v := name v; `(Statement.store $p $v (some $m))
  | `(tritonStmt| tl.store($p:tritonExpr, $v:ident)) => do
    let p ← pointer p; let v := name v; `(Statement.store $p $v none)
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:ident, mask=$m:tritonExpr)) => do
    let p ← pointer p; let m ← test m; let v := name v; `(Statement.atomic $p $v $m)
  | `(tritonStmt| for $i:ident in range(0, $hi:tritonExpr) { $body:tritonStmt* }) => do
    let i := name i; let hi ← expr hi; let body ← body.mapM stmt
    `(Statement.range $i (Expr.literal 0) $hi [$body,*])
  | `(tritonStmt| for $i:ident in range($lo:tritonExpr, $hi:tritonExpr) { $body:tritonStmt* }) => do
    let i := name i; let lo ← expr lo; let hi ← expr hi; let body ← body.mapM stmt
    `(Statement.range $i $lo $hi [$body,*])
  | `(tritonStmt| while $cond:tritonExpr { $body:tritonStmt* }) => do
    let c ← test cond; let body ← body.mapM stmt; `(Statement.loop $c [$body,*])
  | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => do
    let c ← test cond; let yes ← yes.mapM stmt; let no ← no.mapM stmt
    `(Statement.branch $c [$yes,*] [$no,*])
  | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* }) => do
    let c ← test cond; let yes ← yes.mapM stmt; `(Statement.branch $c [$yes,*] [])
  | _ => Macro.throwErrorAt s "unsupported stream-K source statement"

macro_rules
  | `(streamk_triton { $body:tritonStmt* }) => do
    let body ← body.mapM stmt
    `(([$body,*] : List Statement))
end VeriTile.Bench.Repairs.StreamK.TileChecked.Syntax
