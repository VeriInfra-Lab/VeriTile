/-
VeriTile.Triton.DSL.Metadata

Syntax-level metadata collection for `triton { ... }` blocks.
-/

import VeriTile.Triton.DSL.Syntax

open Lean

namespace VeriTile.Triton.DSL.Metadata

/-- Assigned registers and the external regions of their statically visible
pointer values. An empty list still marks a register as assigned. -/
private abbrev RegionBindings := List (String × List (TSyntax `term))

private def methodCastExpr? (stx : TSyntax `tritonExpr) : Option (TSyntax `tritonExpr) :=
  let k := stx.raw.getKind
  if k == ``tritonMethodCast then
    let args := stx.raw.getArgs
    if h : args.size = 5 then
      some ⟨args[0]⟩
    else if h : args.size = 6 then
      some ⟨args[0]⟩
    else
      none
  else
    none

mutual

/-- Collect region terms from statically visible pointer expressions. -/
private partial def staticPtrRegions (assigned : RegionBindings) :
    TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match stx with
  | `(tritonExpr| $(($_:term : Nat))) => []
  | `(tritonExpr| $(($_:term : Int))) => []
  | `(tritonExpr| $(($_:term : Bool))) => []
  | `(tritonExpr| $(($_:term : ℝ))) => []
  | `(tritonExpr| $(($_:term : Real))) => []
  | `(tritonExpr| $($r:term)) => [r]
  | `(tritonExpr| $r:ident) =>
      match assigned.lookup r.getId.toString with
      | some regions => regions
      | none => [⟨r.raw⟩]
  | `(tritonExpr| ($e:tritonExpr)) => staticPtrRegions assigned e
  | `(tritonExpr| $a:tritonExpr + $_b:tritonExpr) =>
      staticPtrRegions assigned a
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $_:ident=$_:tritonExpr,
        $_:ident=[$_:tritonExpr,*], $_:ident=[$_:tritonExpr,*],
        $_:ident=[$_:tritonExpr,*], $_:ident=[$_:tritonExpr,*])) =>
      staticPtrRegions assigned p
  | `(tritonExpr| tl.advance($p:tritonExpr, [$_:tritonExpr,*])) =>
      staticPtrRegions assigned p
  | _ => []

/-- Collect all region terms reachable from a `tritonExpr`. Returns `term`
    syntax - each element is the Lean term inside a `tl.load(...)`
    pointer (recursively in subexpressions). -/
private partial def exprRegions (assigned : RegionBindings) :
    TSyntax `tritonExpr → List (TSyntax `term) := fun stx =>
  match methodCastExpr? stx with
  | some e => exprRegions assigned e
  | none =>
  match stx with
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      staticPtrRegions assigned p ++ exprRegions assigned mask ++ kwargRegions
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      staticPtrRegions assigned p ++ kwargRegions
  | `(tritonExpr| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned v ++ kwargRegions ++ staticPtrRegions assigned p
  | `(tritonExpr| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
            match kw with
            | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned cmp ++ exprRegions assigned v ++ kwargRegions ++ staticPtrRegions assigned p
  | `(tritonExpr| tl.rand($seed:tritonExpr, $offsets:tritonExpr)) =>
      exprRegions assigned seed ++ exprRegions assigned offsets
  | `(tritonExpr| tl.rand($seed:tritonExpr, $offsets:tritonExpr, $rounds:tritonExpr))
  | `(tritonExpr| tl.rand($seed:tritonExpr, $offsets:tritonExpr, n_rounds=$rounds:tritonExpr)) =>
      exprRegions assigned seed ++ exprRegions assigned offsets ++ exprRegions assigned rounds
  | `(tritonExpr| tl.__rand_profiled($_cfg:term, $seed:tritonExpr, $offsets:tritonExpr)) =>
      exprRegions assigned seed ++ exprRegions assigned offsets
  | `(tritonExpr| $_helper:ident($argument:tritonExpr)) => exprRegions assigned argument
  | `(tritonExpr| tl.exp($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| libdevice.exp($e:tritonExpr))  => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.exp($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| libdevice.log($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.log($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| libdevice.expm1($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.expm1($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| libdevice.log1p($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.log1p($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.exp2($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.math.exp2($e:tritonExpr))   => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.pow($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.log($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.log2($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.math.log2($e:tritonExpr))   => exprRegions assigned e
  | `(tritonExpr| tl.sigmoid($e:tritonExpr))     => exprRegions assigned e
  | `(tritonExpr| silu($e:tritonExpr))           => exprRegions assigned e
  | `(tritonExpr| tl.sqrt($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.math.rsqrt($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.rsqrt($e:tritonExpr))      => exprRegions assigned e
  | `(tritonExpr| tl.tanh($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| triton_tanh($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| libdevice.tanh($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.tanh($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tanh($e:tritonExpr))           => exprRegions assigned e
  | `(tritonExpr| tl.sin($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.math.sin($e:tritonExpr))    => exprRegions assigned e
  | `(tritonExpr| tl.cos($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.tan($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.atan($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.cosh($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.sinh($e:tritonExpr))        => exprRegions assigned e
  | `(tritonExpr| tl.erf($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.erf($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.abs($e:tritonExpr))         => exprRegions assigned e
  | `(tritonExpr| tl.logical_and($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.logical_not($e:tritonExpr)) => exprRegions assigned e
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| _div_up($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| min($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr))   =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.cumsum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.cumprod($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.associative_scan($e:tritonExpr, $_:tritonScanOp $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.argmax($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.argmin($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.sort($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.toReal($e:tritonExpr))      => exprRegions assigned e
  | `(tritonExpr| tl.cast($e:tritonExpr, $_:tritonDType)) => exprRegions assigned e
  | `(tritonExpr| tl.extra.cuda.libdevice.llrint($e:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.finitef($e:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.isfinited($e:tritonExpr))
  | `(tritonExpr| libdevice.finitef($e:tritonExpr))
  | `(tritonExpr| libdevice.isfinited($e:tritonExpr)) =>
      exprRegions assigned e
  | `(tritonExpr| tl.multiple_of($e:tritonExpr, $align:tritonExpr)) =>
      exprRegions assigned e ++ exprRegions assigned align
  | `(tritonExpr| tl.max_contiguous($e:tritonExpr, $align:tritonExpr)) =>
      exprRegions assigned e ++ exprRegions assigned align
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned a ++ exprRegions assigned b ++ kwargRegions
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $c:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b ++ exprRegions assigned c
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $_:ident=$base:tritonExpr,
        $_:ident=[$_parent:tritonExpr,*], $_:ident=[$_strides:tritonExpr,*],
        $_:ident=[$_offsets:tritonExpr,*], $_:ident=[$_block:tritonExpr,*])) =>
      staticPtrRegions assigned p ++ exprRegions assigned base
  | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*))) =>
      staticPtrRegions assigned p
  | `(tritonExpr| tl.make_block_ptr($_:ident=$p:tritonExpr,
        $_:ident=($_parent:tritonExpr,*), $_:ident=($_strides:tritonExpr,*),
        $_:ident=($_offsets:tritonExpr,*), $_:ident=($_block:tritonExpr,*),
        $_:ident=($_order:num,*))) =>
      staticPtrRegions assigned p
  | `(tritonExpr| tl.advance($p:tritonExpr, [$_deltas:tritonExpr,*])) =>
      exprRegions assigned p
  | `(tritonExpr| tl.advance($p:tritonExpr, $_:ident=($_deltas:tritonExpr,*))) =>
      exprRegions assigned p
  | `(tritonExpr| tl.permute($e:tritonExpr, [$_:num,*])) =>
      exprRegions assigned e
  | `(tritonExpr| tl.reshape($e:tritonExpr, [$dims:tritonExpr,*])) =>
      dims.getElems.foldl (fun acc d => acc ++ exprRegions assigned d) (exprRegions assigned e)
  | `(tritonExpr| tl.view($e:tritonExpr, [$dims:tritonExpr,*])) =>
      dims.getElems.foldl (fun acc d => acc ++ exprRegions assigned d) (exprRegions assigned e)
  | `(tritonExpr| tl.ravel($e:tritonExpr)) =>
      exprRegions assigned e
  | `(tritonExpr| tl.flip($e:tritonExpr, $_:num)) =>
      exprRegions assigned e
  | `(tritonExpr| tl.flip($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      exprRegions assigned e ++ kwargRegions
  | `(tritonExpr| tl.join($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.split($e:tritonExpr, $_:num)) =>
      exprRegions assigned e
  | `(tritonExpr| ($e:tritonExpr))               => exprRegions assigned e
  | `(tritonExpr| tl.program_id($e:tritonExpr))  => exprRegions assigned e
  | `(tritonExpr| tl.arange($e:tritonExpr))      => exprRegions assigned e
  | `(tritonExpr| tl.arange($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr <  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr >  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr &  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr ^  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr |  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| ~$e:tritonExpr) => exprRegions assigned e
  | `(tritonExpr| not $e:tritonExpr) => exprRegions assigned e
  | `(tritonExpr| -$e:tritonExpr) => exprRegions assigned e
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr +  $b:tritonExpr) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr -  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr *  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr /  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| $a:tritonExpr %  $b:tritonExpr) => exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.where($c:tritonExpr, $a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned c ++ exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.fma($a:tritonExpr, $b:tritonExpr, $c:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b ++ exprRegions assigned c
  | `(tritonExpr| $e:tritonExpr[ : ])          => exprRegions assigned e
  | `(tritonExpr| $e:tritonExpr[ : , None ])     => exprRegions assigned e
  | `(tritonExpr| $e:tritonExpr[ None , : ])     => exprRegions assigned e
  | `(tritonExpr| $e:tritonExpr[ : , None , : ]) => exprRegions assigned e
  | `(tritonExpr| $e:tritonExpr[ None , : , : ]) => exprRegions assigned e
  | `(tritonExpr| $e:tritonExpr[ : , : , None ]) => exprRegions assigned e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:tritonReduceKwarg)) => exprRegions assigned e
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $_:num)) => exprRegions assigned e
  | `(tritonExpr| tl.trans($e:tritonExpr))       => exprRegions assigned e
  | `(tritonExpr| tl.broadcast($a:tritonExpr, $b:tritonExpr)) =>
      exprRegions assigned a ++ exprRegions assigned b
  | `(tritonExpr| tl.broadcast_to($e:tritonExpr, [$_dims:tritonExpr,*])) =>
      exprRegions assigned e
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $v:tritonExpr)) => exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $v:tritonExpr, $_name:ident=$_dt:tritonDType)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $v:tritonExpr,
      $_name:ident=$_:ident.$_:ident.$_:ident)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $_dtypeName:ident=$_dt:tritonDType,
      $_valueName:ident=$v:tritonExpr)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $_valueName:ident=$v:tritonExpr,
      $_dtypeName:ident=$_dt:tritonDType)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $_dtypeName:ident=$_:ident.$_:ident.$_:ident,
      $_valueName:ident=$v:tritonExpr)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.full([$_dims:tritonExpr,*], $_valueName:ident=$v:tritonExpr,
      $_dtypeName:ident=$_:ident.$_:ident.$_:ident)) =>
      exprRegions assigned v
  | `(tritonExpr| tl.zeros([$_dims:tritonExpr,*])) => []
  | `(tritonExpr| tl.zeros([$_dims:tritonExpr,*], $_name:ident=$_dt:term)) => []
  | `(tritonExpr| tl.zeros([$_dims:tritonExpr,*], $_name:ident=$_dt:tritonDType)) => []
  | `(tritonExpr| tl.zeros_like($e:tritonExpr)) => exprRegions assigned e
  | _ => []
end

private def memKwargRegions (assigned : RegionBindings)
    (kwargs : TSyntaxArray `tritonMemKwarg) : List (TSyntax `term) :=
  kwargs.foldl
    (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonMemKwarg) =>
      match kw with
      | `(tritonMemKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
      | _ => acc) []

private def bindRegister (assigned : RegionBindings) (name : Ident)
    (regions : List (TSyntax `term) := []) : RegionBindings :=
  (name.getId.toString, regions) :: assigned

/-- Metadata includes either control-flow path without treating a register
name as an external region. This does not establish runtime boundness. -/
private def joinBindings (left right : RegionBindings) : RegionBindings :=
  ((left.map Prod.fst ++ right.map Prod.fst).eraseDups).map fun name =>
    (name, ((left.lookup name).getD [] ++ (right.lookup name).getD []).eraseDups)

mutual

private partial def blockRegionsWith (assigned : RegionBindings)
    (stmts : List (TSyntax `tritonStmt)) :
    List (TSyntax `term) × List (TSyntax `term) × RegionBindings :=
  stmts.foldl
    (fun (acc : List (TSyntax `term) × List (TSyntax `term) × RegionBindings) st =>
      let (i, o, nextAssigned) := stmtRegionsWith acc.2.2 st
      (acc.1 ++ i, acc.2.1 ++ o, nextAssigned))
    ([], [], assigned)

/-- Per-statement region split plus the assigned-register context after the statement. -/
private partial def stmtRegionsWith (assigned : RegionBindings)
    (stx : TSyntax `tritonStmt) :
    List (TSyntax `term) × List (TSyntax `term) × RegionBindings :=
 match stx with
  | `(tritonStmt| tl.param $name:ident : $_:tritonDType (carrier=$_:tritonDType)) =>
      ([], [], bindRegister assigned name)
  | `(tritonStmt| tl.param $name:ident : $_:tritonDType) =>
      ([], [], bindRegister assigned name)
  | `(tritonStmt| $r0:ident, $r1:ident, $r2:ident, $r3:ident =
      tl.philox($seed:tritonExpr, $c0:tritonExpr, $c1:tritonExpr, $c2:tritonExpr, $c3:tritonExpr)) =>
      let ins := [seed, c0, c1, c2, c3].flatMap (exprRegions assigned)
      let nextAssigned := [r0, r1, r2, r3].foldl (fun acc i => bindRegister acc i) assigned
      (ins, [], nextAssigned)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident = tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      let nextAssigned := bindRegister (bindRegister assigned lhs0) lhs1
      (exprRegions assigned e ++ kwargRegions, [], nextAssigned)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident := tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let kwargRegions : List (TSyntax `term) :=
        kwargs.foldl
          (fun (acc : List (TSyntax `term)) (kw : TSyntax `tritonReduceKwarg) =>
            match kw with
            | `(tritonReduceKwarg| $_:ident = $val:tritonExpr) => acc ++ exprRegions assigned val
            | _ => acc) []
      let nextAssigned := bindRegister (bindRegister assigned lhs0) lhs1
      (exprRegions assigned e ++ kwargRegions, [], nextAssigned)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* = $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let ins := rhs.foldl (fun acc e => acc ++ exprRegions assigned e) []
      let nextAssigned := (lhs.zip rhs).foldl (fun acc (i, e) =>
        bindRegister acc i (staticPtrRegions assigned e)) assigned
      (ins, [], nextAssigned)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* := $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) =>
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      let ins := rhs.foldl (fun acc e => acc ++ exprRegions assigned e) []
      let nextAssigned := (lhs.zip rhs).foldl (fun acc (i, e) =>
        bindRegister acc i (staticPtrRegions assigned e)) assigned
      (ins, [], nextAssigned)
  | `(tritonStmt| $_i:ident % $e:tritonExpr) =>
      (exprRegions assigned e, [], assigned)
  | `(tritonStmt| $i:ident := tl.max($e:tritonExpr, $_n:num)) =>
      (exprRegions assigned e, [], bindRegister assigned i)
  | `(tritonStmt| $i:ident = tl.max($e:tritonExpr, $_n:num)) =>
      (exprRegions assigned e, [], bindRegister assigned i)
  | `(tritonStmt| $i:ident := $e:tritonExpr) =>
      (exprRegions assigned e, [], bindRegister assigned i (staticPtrRegions assigned e))
  | `(tritonStmt| $i:ident = $e:tritonExpr) =>
      (exprRegions assigned e, [], bindRegister assigned i (staticPtrRegions assigned e))
  | `(tritonStmt| $i:ident += $e:tritonExpr) =>
      (exprRegions assigned e, [], bindRegister assigned i
        ((assigned.lookup i.getId.toString).getD []))
  | `(tritonStmt| $i:ident -= $e:tritonExpr) =>
      (exprRegions assigned e, [], bindRegister assigned i
        ((assigned.lookup i.getId.toString).getD []))
  | `(tritonStmt| $i:ident *= $e:tritonExpr) =>
      (exprRegions assigned e, [], bindRegister assigned i)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ exprRegions assigned mask ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $kw0:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
      let kwargs' := #[kw0] ++ kwargs
      (exprRegions assigned v ++ memKwargRegions assigned kwargs',
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr)) =>
      (exprRegions assigned v,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_max($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_min($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_and($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_or($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_xor($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (exprRegions assigned cmp ++ exprRegions assigned v ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned p, assigned)
  | `(tritonStmt| tl.async_copy($dst:tritonExpr, $src:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      (staticPtrRegions assigned src ++ exprRegions assigned src ++ memKwargRegions assigned kwargs,
        staticPtrRegions assigned dst, assigned)
  | `(tritonStmt| tl.async_wait()) =>
      ([], [], assigned)
  | `(tritonStmt| tl.debug_barrier()) =>
      ([], [], assigned)
  | `(tritonStmt| tl.static_print($_msg:term)) =>
      ([], [], assigned)
  | `(tritonStmt| tl.for $idx:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| tl.for $idx:ident in $_:num { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| for $idx:ident in range(0, $($_:term), $($_:term)) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| for $idx:ident in range($stop:tritonExpr) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (exprRegions assigned stop ++ i, o, assigned)
  | `(tritonStmt| for $idx:ident in range($($_:term), $($_:term), $($_:term)) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| for $idx:ident in range($start:tritonExpr, $stop:tritonExpr) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (exprRegions assigned start ++ exprRegions assigned stop ++ i, o, assigned)
  | `(tritonStmt| for $idx:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (exprRegions assigned start ++ exprRegions assigned stop ++ exprRegions assigned step ++ i, o, assigned)
  | `(tritonStmt| for $idx:ident in tl.static_range($($_:term)) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| for $idx:ident in tl.static_range($_:num) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| tl.static_range $idx:ident in $($_:term) { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| tl.static_range $idx:ident in $_:num { $stmts:tritonStmt* }) =>
      let (i, o, _) := blockRegionsWith (bindRegister assigned idx) stmts.toList
      (i, o, assigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* })
  | `(tritonStmt| if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) =>
      let (thenIns, thenOuts, thenAssigned) := blockRegionsWith assigned thenStmts.toList
      let (elseIns, elseOuts, elseAssigned) := blockRegionsWith assigned elseStmts.toList
      -- Metadata inventories all possible branch sources. A name assigned in
      -- either arm is a program register, not a new external region parameter.
      -- This may-assigned union does not make the register definitely bound:
      -- the unchanged interpreter still fails on an unbound runtime path.
      (exprRegions assigned cond ++ thenIns ++ elseIns, thenOuts ++ elseOuts,
        joinBindings thenAssigned elseAssigned)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* })
  | `(tritonStmt| if $cond:tritonExpr { $stmts:tritonStmt* }) =>
      let (ins, outs, bodyAssigned) := blockRegionsWith assigned stmts.toList
      (exprRegions assigned cond ++ ins, outs, joinBindings bodyAssigned assigned)
  | _ => ([], [], assigned)

end

/-- Per-statement region split: `(input regions, output regions)`. -/
partial def stmtRegions (stx : TSyntax `tritonStmt) :
    List (TSyntax `term) × List (TSyntax `term) :=
  let (i, o, _) := stmtRegionsWith [] stx
  (i, o)

/-- Per-block region split, tracking assigned registers so bare pointer
identifiers can be distinguished from pointer registers. -/
partial def blockRegions (stmts : List (TSyntax `tritonStmt)) :
    List (TSyntax `term) × List (TSyntax `term) :=
  let (i, o, _) := blockRegionsWith [] stmts
  (i, o)

end VeriTile.Triton.DSL.Metadata
