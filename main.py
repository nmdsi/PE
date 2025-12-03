from lark import Lark
from lark.indenter import Indenter

class PyIndenter(Indenter):
    NL_type = "_NEWLINE"
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    OPEN_PAREN_types  = ["LPAR", "LSQB", "LBRACE"]   
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]   
    tab_len = 8

grammar = r"""
?start: (_NEWLINE | statement)+


?statement: simple_stmt _NEWLINE
          | if_stmt
          | while_stmt
          | for_stmt
          | func_def
          | import_stmt _NEWLINE


?simple_stmt: assign
            | aug_assign
            | expr_stmt
            | return_stmt
            | pass_stmt
            
lvalue: NAME ltrailer*
ltrailer: "." NAME                -> attr_tr
        | LSQB [expr ("," expr)*] RSQB -> index_tr


assign:    lvalue "=" expr
aug_assign: lvalue AUGOP expr
expr_stmt: expr
return_stmt: "return" expr
pass_stmt: "pass"

block: ":" _NEWLINE _INDENT (_NEWLINE | statement)+ _DEDENT

import_stmt: import_plain | import_from
import_plain: "import" module_path ["as" NAME]
import_from:  "from" module_path "import" NAME ["as" NAME]
module_path: NAME ("." NAME)*
?expr: conditional
?conditional: or_expr ["if" or_expr "else" conditional]
?or_expr: and_expr ( "|" and_expr )*
?and_expr: comparison ( "&" comparison )*
?comparison: arith ( (COMP | "in") arith )*
?arith: term
      | arith OP_ADD term    -> binop
?term: unary
     | term OP_MUL unary     -> binop

?unary: "-" unary            -> neg_expr
     | "not" unary           -> not_expr
     | postfix
tuple_lit: LPAR expr ("," expr)+ RPAR

?postfix: atom trailer*
?atom: NAME
     | SIGNED_NUMBER
     | ESCAPED_STRING
     | FSTRING
     | literal_list
     | literal_dict
     | tuple_lit
     | LPAR expr RPAR

?trailer: "." NAME                        -> attr_tr
        | LPAR [arg ("," arg)*] RPAR      -> call_tr
        | LSQB [expr ("," expr)*] RSQB    -> index_tr

dict_pair: expr ":" expr

literal_dict: LBRACE [dict_pair ("," dict_pair)*] RBRACE

?arg: expr | NAME "=" expr

literal_list: LSQB [expr ("," expr)*] RSQB

if_stmt: "if" expr block ("elif" expr block)* ["else" block]

while_stmt: "while" expr block
for_stmt:   "for" NAME "in" expr block
func_def:   "def" NAME LPAR [NAME ("," NAME)*] RPAR block

OP_MUL:  "*"|"/"
OP_ADD:  "+"|"-"
AUGOP:   "+="|"-="|"*="|"/="
COMP:    "=="|"!="|">"|"<"|">="|"<="

LPAR: "("
RPAR: ")"
LSQB: "["
RSQB: "]"
LBRACE: "{"
RBRACE: "}"

_NEWLINE: /(\r?\n[ \t]*)+/

FSTRING.2: /[fF](\"([^\"\\\n]|\\.)*\"|'([^'\\\n]|\\.)*')/
COMMENT: /#[^\n]*/      
%import common.CNAME           -> NAME
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS_INLINE
%ignore WS_INLINE
%ignore COMMENT
%declare _INDENT _DEDENT
"""
parser = Lark(grammar, parser="lalr", postlex=PyIndenter(), lexer="basic")
code = """
import pandas as pd

def inc(x):
    return x + 1
def square(x):
    return x * x
def cube(x):
    return x * x * x
def test1(a, b):
    return a * 10 + b
def foo(a, b):
    pass

#NON-RESIDUAL examples
CONST_K   = STATIC(3)                           # compile-time constant
EVENT_MAP = STATIC(dict([("Mortality", "M"), ("B", "B")]))# compile-time lookup


df = pd.read_csv("m3d1.csv")
df = df[df["YEAR"].eq(2020)]
df = df[df["EVENT_TYPE"].eq("Mortality")]

df1 = pd.read_csv("m3d2.csv")

df["rate"] = df1["COUNT"] / df["POPULATION"]
df["tripled_count"] = df1["COUNT"] * CONST_K     # uses static CONST_K; line itself is residual

if df1["COUNT"].eq(2).any():
    df_flag = df1.loc[df1["COUNT"] > 0].copy()
else:
    df_flag = df1.copy()

while df_flag["COUNT"].eq(2).any():
    df_flag.loc[df_flag["COUNT"].eq(2), "COUNT"] += 1

for col_name in ["YEAR", "EVENT_TYPE"]:
    df_flag[col_name] = df_flag[col_name] * 2

for i in range(3):
    df_flag[f"N{i}"] = df_flag["COUNT"] + i

for one in ["SINGLE"]:
    if one in df_flag:
        df_flag[one] = df_flag[one].apply(inc)

df_flag["x2"]   = df_flag["COUNT"].apply(inc)
df_flag["sq"]   = df_flag["COUNT"].apply(square)
df_flag["cube"] = df_flag["COUNT"].apply(cube)

for i in range(2):
    df[f"Z{i}"] = df["POPULATION"] + i

def scale_and_tag(df_in, scale, tag):
    for col_name in ["YEAR", "EVENT_TYPE"]:
        df_in[col_name] = df_in[col_name] * scale
    # -------- NON-RESIDUAL example
    df_in["TAG"] = STATIC("T" + str(tag))                  # RHS computed now; prints as literal e.g. "T7"

scale_and_tag(df_flag, 2, 7)

if "POPULATION" in df_flag:
    df_flag["score"] = df_flag["POPULATION"].apply(square)

foo(1, 2)

# FORCED-RESIDUAL 
df = DYNAMIC(df.assign(ZZ=1))                   # will print as: df = df.assign(ZZ=1)

for i in range(2):
    df[f"tag{i}"] = f"T{i}"

s = df_flag["COUNT"]
s2 = s.apply(inc).apply(square)

col = df_flag["EVENT_TYPE"] if "EVENT_TYPE" in df_flag else pd.Series(dtype=object)
m = col.isin(["A"]) | col.isin(["B", "C"])

if "PATH" in df_flag:
    t = df_flag["PATH"].str.split("/")
    t = t.str[1]

if "EVENT_TYPE" in df_flag:
    while df_flag["EVENT_TYPE"].eq("X").any():
        df_flag.loc[df_flag["EVENT_TYPE"].eq("X"), "EVENT_TYPE"] += "!"

# Already static branch (won't be residualized)
if 1 < 2:
    z = 10
else:
    z = 20


TOTAL = 0



"""

tree = parser.parse(code)
#print(tree)
from lark import Token, Transformer
class ASTBuilder(Transformer):
    
    def comparison(self, items):
        if not items:
            return ("var", "__empty_comparison__")
    
        acc = items[0]
        i = 1
    
        while i + 1 < len(items):
            op = items[i]
            rhs = items[i + 1]
            op = getattr(op, "value", op)  
            acc = ("compare", acc, op, rhs)
            i += 2

        if i < len(items):
            rhs = items[i]
            acc = ("compare", acc, "in", rhs)
    
        return acc
        

    def tuple_lit(self, items):
        return ("literal_tuple", items)

    
    def dict_pair(self, items):
        return ("pair", items[0], items[1])
    
    def literal_dict(self, items):
        return ("literal_dict", items)
    
    def module_path(self, items):
        parts = []
        for it in items:
            if isinstance(it, tuple) and it[0] == "var":
                parts.append(it[1])
            else:
                parts.append(getattr(it, "value", str(it)))
        return ".".join(parts)
    
    def import_plain(self, items):
        module = items[0] if isinstance(items[0], str) else str(items[0])
        alias = items[1][1] if len(items) > 1 and isinstance(items[1], tuple) else None
        return ("import_plain", module, alias)
    
    def import_from(self, items):
        module = items[0] if isinstance(items[0], str) else str(items[0])
        name = items[1][1] if isinstance(items[1], tuple) else str(items[1])
        alias = items[2][1] if len(items) > 2 and isinstance(items[2], tuple) else None
        return ("import_from", module, name, alias)
    
    def import_stmt(self, items):
        return items[0]
    

    

    def literal_list(self, items):
        exprs = [x for x in items if not isinstance(x, Token)]
        return ("literal_list", exprs)


 
    
    def pass_stmt(self, _):
        return ("pass_stmt",)

    def lvalue(self, items):
        base = items[0]
        for tr in items[1:]:
            if not (isinstance(tr, tuple) and tr and tr[0] == "__trailer__"):
                continue
            kind = tr[1][0]
            payload = tr[1][1:]
    
            if kind == "attr":
                (name,) = payload
                base = ("attr_of", base, name)
    
            elif kind == "index":
                (idxs,) = payload
    
                if (isinstance(base, tuple) and base[0] == "attr_of"
                    and base[2] == "loc" and len(idxs) == 2):
                    rows, cols = idxs[0], idxs[1]
                    return ("loc_lhs", base[1], rows, cols)
    
                idx = idxs[0] if len(idxs) == 1 else ("literal_list", idxs)
    
                if isinstance(base, tuple) and base[0] == "var":
                    base = ("col_access", base, idx)
                else:
                    base = ("index_get", base, idx)
        return base

    def column_assign(self, items):
        var_node, idx, rhs = items
        var = var_node[1] if isinstance(var_node, tuple) else var_node
        return ("col_op", var, idx, rhs)

    def read_csv(self, items):
        s = items[0]
        fname = s[1] if isinstance(s, tuple) else str(s)[1:-1]
        return ("read_csv", fname)

    def filter_expr(self, items):
        return ("filter", items[0], items[1])

    def binop(self, items):
        left, op, right = items
        op = getattr(op, "value", op)
        return ("binop", left, op, right)

    def str_method_call(self, items):
        recv, name_tok, *args = items
        if isinstance(name_tok, Token):
            method_name = name_tok.value
        elif isinstance(name_tok, tuple) and name_tok[0] == "var":
            method_name = name_tok[1]
        else:
            method_name = str(name_tok)
        return ("str_method_call", recv, method_name, args)

    def str_index(self, items):
        recv, _, idx = items
        return ("str_get", recv, idx)

    def col_access(self, items):
        var_node, idx = items
        var = var_node[1] if isinstance(var_node, tuple) else var_node
        return ("col_access", ("var", var) if not isinstance(var_node, tuple) else var_node, idx)

    def func_call(self, items):
        name = items[0]
        args = items[1:]
        return ("func_call", name, args)

    def expr_stmt(self, items):
        return items[0]

    def block(self, items):
        return list(items)

    def if_stmt(self, items):
        cond, then_block, *rest = items
        else_block = rest[0] if rest else []
        return ("if", cond, then_block, else_block)

    def while_stmt(self, items):
        cond, body = items
        return ("while", cond, body)

    def for_stmt(self, items):
        var, iterable, body = items
        return ("for", var, iterable, body)

    def func_def(self, items):
        name = items[0]
        body = items[-1]
        params = [p for p in items[1:-1] if isinstance(p, tuple) and p[0] == "var"]
        return ("func_def", name, params, body)

    def var(self, items):
        return ("var", str(items[0]))

    def number(self, items):
        return ("number", float(items[0]))

    def string(self, items):
        return ("string", items[0][1:-1])

    def dotted_name(self, items):
        parts = []
        for it in items:
            if isinstance(it, tuple):
                parts.append(it[1])
            else:
                parts.append(getattr(it, "value", str(it)))
        return ".".join(parts)

    def statement(self, items):
        return items[0]

    def start(self, items):
        return items

    def NAME(self, tok):
        return ("var", tok.value)

    def COMP(self, tok):
        return tok.value

    def ESCAPED_STRING(self, tok):
        return ("string", tok[1:-1])

    def SIGNED_NUMBER(self, tok):
        return ("number", float(tok))

    def compare(self, items):
        left, op, lit = items
        op = getattr(op, "value", op)
        if isinstance(lit, Token):
            if lit.type == 'SIGNED_NUMBER':
                lit_node = ("number", float(lit))
            else:
                lit_node = ("string", lit[1:-1])
        else:
            lit_node = lit
        return ("compare", left, op, lit_node)

    def neg_expr(self, items):
        return ("unop", "-", items[0])

    def not_expr(self, items):
        return ("unop", "not", items[0])

    def return_stmt(self, items):
        return ("return", items[0])

    def method_call(self, items):
        receiver, name_tok, *args = items
        if isinstance(name_tok, Token):
            method_name = name_tok.value
        elif isinstance(name_tok, tuple) and name_tok[0] == "var":
            method_name = name_tok[1]
        else:
            method_name = str(name_tok)
        return ("method_call", receiver, method_name, args)
    def conditional(self, items):
       
        if len(items) == 1:
            return items[0]
        if len(items) == 3 and items[1] is None and items[2] is None:
            return items[0]
        then_e, if_e, else_e = items
        return ("cond", if_e, then_e, else_e)

    def arg(self, items):
        if len(items) == 1:
            return items[0]
        name, value = items[0], items[1]
        key = name[1] if (isinstance(name, tuple) and name and name[0] == "var") else str(getattr(name, "value", name))
        return ("kwarg", key, value)

    def attr_tr(self, items):  
        (name,) = items
        key = name[1] if (isinstance(name, tuple) and name and name[0] == "var") else str(getattr(name, "value", name))
        return ("__trailer__", ("attr", key))

    def call_tr(self, items):
        args = []
        for a in items:
            if isinstance(a, Token) and a.type in ("LPAR", "RPAR", "COMMA"):
                continue
            if a is not None:
                args.append(a)
        return ("__trailer__", ("call", args))

    def index_tr(self, items):
        exprs = []
        for it in items:
            if isinstance(it, Token) and it.type in ("LSQB", "RSQB", "COMMA"):
                continue
            exprs.append(it)
        return ("__trailer__", ("index", exprs))

    
    def or_expr(self, items):
        acc = items[0]
        for rhs in items[1:]:
            acc = ("binop", acc, "|", rhs)
        return acc



    def assign(self, items):
        lhs, rhs = items
        if isinstance(lhs, tuple):
            if lhs[0] == "col_access":
                var_node, idx = lhs[1], lhs[2]
                return ("col_op", var_node, idx, rhs)
            if lhs[0] == "index_get" and isinstance(lhs[1], tuple) and lhs[1][0] == "attr_of" and lhs[1][2] == "loc":
                df_node = lhs[1][1]
                rows, cols = lhs[2][1] if (isinstance(lhs[2], tuple) and lhs[2][0] == "literal_list") else (lhs[2], None)
                return ("loc_set", df_node, rows, cols, rhs)
        return ("assign", lhs, rhs)
    
    def simple_assign(self, items):
        lhs, rhs = items
        if isinstance(lhs, tuple):
            if lhs[0] == "col_access":
                var_node, idx = lhs[1], lhs[2]
                return ("col_op", var_node, idx, rhs)
            if lhs[0] == "loc_lhs":
                _, df_node, rows, cols = lhs
                return ("loc_set", df_node, rows, cols, rhs)
        return ("assign", lhs, rhs)

    def aug_assign(self, items):
        lhs, op, rhs = items
        op = str(op)
        if isinstance(lhs, tuple):
            if lhs[0] == "col_access":
                var_node, idx = lhs[1], lhs[2]
                return ("aug_assign", ("col_access", var_node, idx), op, rhs)
            if lhs[0] == "loc_lhs":
                _, df_node, rows, cols = lhs
                idx_list = ("literal_list", [rows, cols])
                lhs_norm = ("index_get", ("attr_of", df_node, "loc"), idx_list)
                return ("aug_assign", lhs_norm, op, rhs)
            if lhs[0] == "index_get" and isinstance(lhs[1], tuple) and lhs[1][0] == "attr_of" and lhs[1][2] == "loc":
                return ("aug_assign", lhs, op, rhs)
        return ("aug_assign", lhs, op, rhs)
    
    def and_expr(self, items):
        acc = items[0]
        for rhs in items[1:]:
            acc = ("binop", acc, "&", rhs)
        return acc
    def postfix(self, items):
        base, *trailers = items

        def is_attr_of(node, name):
            return isinstance(node, tuple) and node[0] == "attr_of" and node[2] == name

        for tr in trailers:
            kind, payload = tr[1][0], tr[1][1:]
            if kind == "attr":
                (name,) = payload
                base = ("attr_of", base, name)

            elif kind == "call":
                    (args,) = payload  
                    if isinstance(base, tuple) and base[0] == "attr_of":
                        recv, name = base[1], base[2]
                        base = ("method_call", recv, name, args)
                    elif isinstance(base, tuple) and base[0] == "var":
                        base = ("func_call", base, args)
                    else:
                        base = ("call", base, args)           
       

            elif kind == "index":
                (parts,) = payload
                idx = parts[0] if len(parts) == 1 else ("literal_list", parts)

                if isinstance(base, tuple) and base[0] == "attr_of" and base[2] == "loc":
                    if len(parts) == 2:
                        rows, cols = parts
                        base = ("index_get", base, ("literal_list", [rows, cols]))
                    else:
                        base = ("index_get", base, idx)
                    continue

                if isinstance(base, tuple) and base[0] == "attr_of" and base[2] == "str":
                    base = ("str_get", base[1], idx)
                    continue

                if isinstance(base, tuple) and base[0] == "var":
                    base = ("col_access", base, idx)
                else:
                    base = ("index_get", base, idx)

        return base

def parse_code(code: str):
    tree = parser.parse(code)
    return ASTBuilder().transform(tree)

def substitute_var(node, varname, replacement):
    if isinstance(node, Token) and getattr(node, "type", "") == "FSTRING":
        s = node.value
        s_clean = s.lstrip("fF")
        if s_clean and s_clean[0] in ('"', "'") and s_clean[-1] == s_clean[0]:
            inner = s_clean[1:-1]
        else:
            inner = s_clean[1:-1] if len(s_clean) >= 2 else s_clean

        def _to_str(v):
            if isinstance(v, tuple):
                if v[0] == "number":
                    x = v[1]
                    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
                if v[0] == "string":
                    return v[1]
            if isinstance(v, (int, float)):
                return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
            return str(v)

        rep = _to_str(replacement)
        new_inner = inner.replace("{"+varname+"}", rep)
        if new_inner != inner:
            return ("string", new_inner)
        return node

    if isinstance(node, tuple) and node and node[0] == "var" and node[1] == varname:
        return replacement
    if isinstance(node, tuple):
        return tuple(substitute_var(c, varname, replacement) for c in node)
    if isinstance(node, list):
        return [substitute_var(c, varname, replacement) for c in node]
    return node
class Lattice:
    STATIC = "static"
    DYNAMIC = "dynamic"
    STATIC_SCHEMA = "schema"

    @staticmethod
    def lub(a, b):
        order = [Lattice.STATIC, Lattice.STATIC_SCHEMA, Lattice.DYNAMIC]
        return max(a, b, key=order.index)


def _s(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)

def fold_constants(expr, interp):
    val, bt = interp.eval(expr)
    if bt == Lattice.STATIC:
        if isinstance(val, str):
            return ("string", val)
        if isinstance(val, (int, float)):
            return ("number", float(val))
    return expr


#loop unrolling 
def _unroll_block(block, const_interp, max_unroll=64):
    out = []
    for st in block:
        if isinstance(st, tuple) and st and st[0] == "for":
            _, var_node, iterable, body = st
            if isinstance(iterable, tuple) and iterable[0] == "literal_list":
                _, lits = iterable
                if len(lits) <= max_unroll:
                    vname = var_node[1] if (isinstance(var_node, tuple) and var_node[0] == "var") else var_node
                    for lit in lits:
                        lit_node = lit if isinstance(lit, tuple) else ("string", lit)
                        for inner in body:
                            inner2 = substitute_var(inner, vname, lit_node)
                            if isinstance(inner2, tuple) and inner2[0] == "col_op":
                                a, b, col, rhs = inner2
                                col = fold_constants(col, const_interp)
                                rhs = fold_constants(rhs, const_interp)
                                inner2 = (a, b, col, rhs)
                            out.append(inner2)
                    continue
        out.append(st)
    return out

def unroll_loops(ast_nodes, max_unroll=8):
    out = []
    const_interp = Interpreter()

    for node in ast_nodes:
        tag = node[0]

        if tag == "for" and node[2][0] == "literal_list":
            _, var_node, (_, lits), body = node
            if len(lits) <= max_unroll:
                vname = var_node[1] if isinstance(var_node, tuple) else var_node
                for lit in lits:
                    replacement = lit if isinstance(lit, tuple) else ("string", lit)
                    for stmt in body:
                        stmt2 = substitute_var(stmt, vname, replacement)
                        if stmt2[0] == "col_op":
                            a, b, col, rhs = stmt2
                            col = fold_constants(col, const_interp)
                            rhs = fold_constants(rhs, const_interp)
                            stmt2 = (a, b, col, rhs)
                        out.append(stmt2)
                continue

        if tag == "for" and node[2][0] == "func_call":
            _, var_node, call_node, body = node
            name_node = call_node[1]
            if isinstance(name_node, tuple) and name_node[0] == "var" and name_node[1] == "range":
                args = call_node[2]
                if len(args) == 1 and args[0][0] == "number" and args[0][1] <= max_unroll:
                    vname = var_node[1] if isinstance(var_node, tuple) else var_node
                    for i in range(int(args[0][1])):
                        lit = ("number", float(i))
                        for stmt in body:
                            stmt2 = substitute_var(stmt, vname, lit)
                            if stmt2[0] == "col_op":
                                a, b, col, rhs = stmt2
                                col = fold_constants(col, const_interp)
                                rhs = fold_constants(rhs, const_interp)
                                stmt2 = (a, b, col, rhs)
                            out.append(stmt2)
                    continue

        out.append(node)

    return out
import pandas as pd

class Interpreter:
    def __init__(self, static_files=None, inlineable=None):
        self.env = {}
        self.bt = {}
        self.residual = []
        self.static_files = set(static_files or [])
        self.inlineable = inlineable or {}
        self.schema: dict[int, dict[str, str]] = {}
        self.spec_index = {}
        self.spec_defs = []
        self._spec_ctr = 0


    def specialize_function(self, fname, params, body, static_pairs, dynamic_pairs):
        key = (fname, tuple(v for (_, v) in static_pairs))
        if key in self.spec_index:
            return self.spec_index[key]

        self._spec_ctr += 1
        spec_name = f"{fname}__S{self._spec_ctr}"
        self.spec_index[key] = spec_name

        has_return = any(st[0] == "return" for st in body)

        if has_return:
            subst_body = body
            for (p, v) in static_pairs:
                lit = ("number", float(v)) if isinstance(v, (int, float)) else ("string", v)
                subst_body = [substitute_var(stmt, p, lit) for stmt in subst_body]
            spec_body = subst_body
        else:
            subst_body = body
            for (p, v) in static_pairs:
                lit = ("number", float(v)) if isinstance(v, (int, float)) else ("string", v)
                subst_body = [substitute_var(stmt, p, lit) for stmt in subst_body]
            subst_body = _unroll_block(subst_body, self)
            old_env, old_bt, old_resid = self.env.copy(), self.bt.copy(), self.residual
            self.residual = []
            for (p, v) in static_pairs:
                self.env[p] = v
                self.bt[p]  = Lattice.STATIC
            for stmt in subst_body:
                self.run([stmt])
            spec_body = self.residual
            self.residual = old_resid
            self.env, self.bt = old_env, old_bt

        dyn_params = [("var", p) for (p, _) in dynamic_pairs]
        self.spec_defs.append(("func_def", ("var", spec_name), dyn_params, spec_body))
        return spec_name

    def run(self, ast_nodes):
        for node in ast_nodes:
            tag = node[0]

            if tag == "assign":
                _, lhs, rhs = node
                if isinstance(lhs, tuple):
                    if lhs[0] == "col_access":
                        var_node, idx = lhs[1], lhs[2]
                        self.run([("col_op", var_node, idx, rhs)])
                        continue
            
                    if lhs[0] == "index_get" and isinstance(lhs[1], tuple) and lhs[1][0] == "attr_of" and lhs[1][2] == "loc":
                        df_node = lhs[1][1]
                        if isinstance(lhs[2], tuple) and lhs[2][0] == "literal_list" and len(lhs[2][1]) == 2:
                            rows, cols = lhs[2][1]
                        else:
                            rows, cols = lhs[2], None
                        self.run([("loc_set", df_node, rows, cols, rhs)])
                        continue
            
                    if lhs[0] == "index_get" and isinstance(lhs[1], tuple) and lhs[1][0] == "var":
                        var_node = lhs[1]
                        idx = lhs[2]
                        self.run([("col_op", var_node, idx, rhs)])
                        continue
            
                var_name = lhs[1] if isinstance(lhs, tuple) and lhs and lhs[0] == "var" else lhs
                val, bt = self.eval(rhs)
                self.env[var_name] = val
                self.bt[var_name] = bt
                if bt == Lattice.DYNAMIC:
                    self.residual.append(node)

            elif tag == "filter":
                _, var, cond = node
                var = var[1] if isinstance(var, tuple) else var
                df, bt_df = self.env[var], self.bt[var]
                mask, bt_mask = self.eval(cond)
                df2 = df[mask]
                if id(df) in self.schema:
                    self.schema[id(df2)] = self.schema[id(df)]
                new_bt = Lattice.lub(bt_df, bt_mask)
                self.env[var] = df2
                self.bt[var] = new_bt
                if new_bt == Lattice.DYNAMIC:
                    self.residual.append(node)

            elif tag == "col_op":
                _, var_node, idx_expr, rhs = node
                var = var_node[1] if isinstance(var_node, tuple) else var_node
                if var not in self.env:
                    col_val, bt_idx = self.eval(idx_expr)
                    folded_idx = ("string", col_val) if bt_idx == Lattice.STATIC else idx_expr
            
                    rhs2 = fold_constants(rhs, self)
            
                    if bt_idx == Lattice.STATIC and isinstance(idx_expr, tuple) and idx_expr[0] == "var":
                        rhs2 = substitute_var(rhs2, idx_expr[1], ("string", col_val))
            
                    self.residual.append(("col_op", ("var", var), folded_idx, rhs2))
                    continue
            
                df, bt_df = self.env[var], self.bt[var]
                col_val, bt_idx = self.eval(idx_expr)
                val, bt_val = self.eval(rhs)
            
                df2 = df.copy()
                if id(df) in self.schema:
                    self.schema[id(df2)] = self.schema[id(df)]
                df2[col_val] = val
                new_bt = Lattice.lub(bt_df, Lattice.lub(bt_idx, bt_val))
                self.env[var] = df2
                self.bt[var]  = new_bt
            
                folded_idx = ("string", col_val) if bt_idx == Lattice.STATIC else idx_expr
            
                if new_bt == Lattice.DYNAMIC:
                    rhs2 = fold_constants(rhs, self)
                    if bt_idx == Lattice.STATIC and isinstance(idx_expr, tuple) and idx_expr[0] == "var":
                        rhs2 = substitute_var(rhs2, idx_expr[1], ("string", col_val))
                    self.residual.append(("col_op", ("var", var), folded_idx, rhs2))

            elif tag == "if":
                _, cond, then_blk, else_blk = node
                val, bt_cond = self.eval(cond)
                if bt_cond == Lattice.STATIC and isinstance(val, (bool, int, float)):
                    branch = then_blk if bool(val) else (else_blk or [])
                    for stmt in branch:
                        self.run([stmt])
                else:
                    self.residual.append(node)

            elif tag == "for":
                _, var, iterable_node, body = node
                var = var[1] if isinstance(var, tuple) else var
                iter_val, bt_iter = self.eval(iterable_node)
                if bt_iter == Lattice.STATIC and hasattr(iter_val, "__iter__"):
                    for item in iter_val:
                        self.env[var], self.bt[var] = item, Lattice.STATIC
                        for stmt in body:
                            self.run([stmt])
                else:
                    self.residual.append(node)

            elif tag == "while":
                _, cond_node, body = node
                val, bt_cond = self.eval(cond_node)
                if bt_cond == Lattice.STATIC and not bool(val):
                    pass
                else:
                    self.residual.append(node)
                    
            elif tag in ("import_plain", "import_from", "pass_stmt"):
                self.residual.append(node)
                continue
                
            elif tag == "func_def":
                _, name, args, body = node
                fname = name[1] if isinstance(name, tuple) else name
                self.env[fname] = ("func", args, body)
                self.bt[fname]  = Lattice.STATIC

            elif tag == "func_call":
                _, name_node, arg_nodes = node
                fname = name_node[1] if isinstance(name_node, tuple) else name_node
                entry = self.env.get(fname)
                if not entry or entry[0] != "func":
                    self.residual.append(node)
                    continue

                params, body = entry[1], entry[2]
                vals_bts = [self.eval(a) for a in arg_nodes]
                pe_args  = [fold_constants(a, self) for a in arg_nodes]

                static_pairs, dynamic_pairs = [], []
                for pnode, ((val, bt), rexpr) in zip(params, zip(vals_bts, pe_args)):
                    pname = pnode[1] if isinstance(pnode, tuple) else pnode
                    if bt == Lattice.STATIC:
                        static_pairs.append((pname, val))
                    else:
                        dynamic_pairs.append((pname, rexpr))

                spec_name = self.specialize_function(fname, params, body, static_pairs, dynamic_pairs)
                dyn_actuals = [rexpr for (_, rexpr) in dynamic_pairs]
                self.residual.append(("func_call", ("var", spec_name), dyn_actuals))

            else:
                raise NotImplementedError(tag)



    def eval(self, node):
            if node is None:
                raise ValueError("eval() received None")
            if isinstance(node, Token):
                return node.value, Lattice.STATIC

            if isinstance(node, (str, int, float, bool)):
                return node, Lattice.STATIC
            if isinstance(node, list):
                vals, bt = [], Lattice.STATIC
                for it in node:
                    v, b = self.eval(it)
                    vals.append(v)
                    bt = Lattice.lub(bt, b)
                return vals, bt
            tag = node[0]
            if tag == "literal_tuple":
                vals = []
                bt = Lattice.STATIC
                for it in node[1]:
                    v, b = self.eval(it)
                    vals.append(v)
                    bt = Lattice.lub(bt, b)
                return tuple(vals), bt
                
            if tag == "literal_dict":
                result = {}
                bt = Lattice.STATIC
                for _, k_node, v_node in node[1]:  
                    kv, kb = self.eval(k_node)
                    vv, vb = self.eval(v_node)
                    bt = Lattice.lub(bt, Lattice.lub(kb, vb))
                    result[kv] = vv
                return result, bt
         
            
    
          
    
            if not isinstance(node, tuple):
                return node, Lattice.DYNAMIC
            if tag == "unop":
                _, op, operand = node
                val, bt = self.eval(operand)
                if bt == Lattice.STATIC:
                    if op == "-":   return -val, Lattice.STATIC
                    if op == "not": return (not val), Lattice.STATIC
                return None, Lattice.DYNAMIC
        

            if tag == "method_call":
                _, recv_ast, name, args_ast = node
                recv_val, recv_bt = self.eval(recv_ast)
            
                pos_vals, kw_vals = [], {}
                args_bt = Lattice.STATIC
                if any(a is None for a in args_ast):
                    raise ValueError(f"AST error: method_call '{name}' received a None argument. "f"recv_ast={recv_ast!r}, args_ast={args_ast!r}")
                
                for a in args_ast:
                    if isinstance(a, tuple) and a[0] == "kwarg":
                        _, k, v_ast = a
                        v, b = self.eval(v_ast)
                        kw_vals[k] = v
                        args_bt = Lattice.lub(args_bt, b)
                    else:
                        v, b = self.eval(a)
                        pos_vals.append(v)
                        args_bt = Lattice.lub(args_bt, b)
            
                if name == "apply":
                    fn_node = args_ast[0] if args_ast else None
                    if isinstance(fn_node, tuple) and fn_node[0] == "var":
                        fn_name = fn_node[1]
                        if fn_name in self.inlineable:
                            param, fn_body = self.inlineable[fn_name]
                            inlined_ast = substitute_var(fn_body, param, recv_ast)
                            return self.eval(inlined_ast)
                    if isinstance(recv_ast, tuple) and recv_ast[0] == "method_call" and recv_ast[2] == "apply":
                        inner_fn = recv_ast[3][0] if recv_ast[3] else None
                        outer_fn = fn_node
                        if all(isinstance(f, tuple) and f[0] == "var" for f in (inner_fn, outer_fn)):
                            comp = ("func_comp", inner_fn, outer_fn)
                            return (("method_call", recv_ast[1], "apply", [comp]), Lattice.DYNAMIC)
                    try:
                        if hasattr(recv_val, "apply") and pos_vals:
                            out = recv_val.apply(pos_vals[0], **kw_vals)
                            return out, Lattice.lub(recv_bt, args_bt)
                    except Exception:
                        return None, Lattice.DYNAMIC
                    return None, Lattice.DYNAMIC
            
                try:
                    fn = getattr(recv_val, name)
                    out = fn(*pos_vals, **kw_vals)
                    return out, Lattice.lub(recv_bt, args_bt)
                except Exception:
                    return None, Lattice.DYNAMIC

            if tag == "str_method_call":
                _, recv_ast, name, args_ast = node
                recv_val, recv_bt = self.eval(recv_ast)
            
                pos_vals, kw_vals = [], {}
                args_bt = Lattice.STATIC
                for a in args_ast:
                    if isinstance(a, tuple) and a[0] == "kwarg":
                        _, k, v_ast = a
                        v, b = self.eval(v_ast)
                        kw_vals[k] = v
                        args_bt = Lattice.lub(args_bt, b)
                    else:
                        v, b = self.eval(a)
                        pos_vals.append(v)
                        args_bt = Lattice.lub(args_bt, b)
            
                try:
                    out = getattr(recv_val.str, name)(*pos_vals, **kw_vals)
                    return out, Lattice.lub(recv_bt, args_bt)
                except Exception:
                    return None, Lattice.DYNAMIC
        
            if tag == "read_csv":
                _, fname = node
                df = pd.read_csv(fname)
                if fname in self.static_files:
                    self.schema[id(df)] = df.dtypes.astype(str).to_dict()
                    return df, Lattice.STATIC_SCHEMA
                return df, Lattice.DYNAMIC
        
            if tag == "filter":
                _, var, cond = node
                var = var[1] if isinstance(var, tuple) else var
                df, bt_df = self.env[var], self.bt[var]
                mask, bt_mask = self.eval(cond)
                new_bt = Lattice.lub(bt_df, bt_mask)
                result = df[mask]
                if id(df) in self.schema:
                    self.schema[id(result)] = self.schema[id(df)]
                return result, new_bt
            if tag == "attr_of":
                _, recv_ast, name = node
                recv_val, recv_bt = self.eval(recv_ast)
                try:
                    return getattr(recv_val, name), recv_bt
                except Exception:
                    return None, Lattice.DYNAMIC
            
            if tag == "index_get":
                _, base_ast, idx_ast = node
                base_val, base_bt = self.eval(base_ast)
                idx_val, idx_bt = self.eval(idx_ast)
                try:
                    out = base_val.__getitem__(idx_val)
                    return out, Lattice.lub(base_bt, idx_bt)
                except Exception:
                    return None, Lattice.DYNAMIC
            if tag == "binop":
                _, left_node, op, right_node = node
                lv, bt_l = self.eval(left_node)
                rv, bt_r = self.eval(right_node)
                bt = Lattice.lub(bt_l, bt_r)
        
                if op in ("|", "&"):
                    try:
                        val = (lv | rv) if op == "|" else (lv & rv)
                        return val, bt  
                    except Exception:
                        return None, Lattice.DYNAMIC
        
                if op == "+" and bt_l == bt_r == Lattice.STATIC:
                    if isinstance(lv, str) or isinstance(rv, str):
                        return _s(lv) + _s(rv), Lattice.STATIC
                if bt_l == Lattice.STATIC and bt_r == Lattice.STATIC:
                    val = {"+": lv + rv, "-": lv - rv, "*": lv * rv, "/": lv / rv}[op]
                    return val, Lattice.STATIC
                return None, Lattice.DYNAMIC
        
        
            if tag == "col_access":
                _, var_node, idx_expr = node
                var_name = var_node[1] if isinstance(var_node, tuple) else var_node
                df_obj = self.env.get(var_name)
                bt_df  = self.bt.get(var_name, Lattice.DYNAMIC)
                if df_obj is None:
                    return None, Lattice.DYNAMIC
        
                if isinstance(idx_expr, tuple) and idx_expr[0] == "literal_list":
                    cols, bt_cols = self.eval(idx_expr)   
                    try:
                        out = df_obj[cols]
                        return out, Lattice.lub(bt_df, bt_cols)
                    except Exception:
                        return None, Lattice.DYNAMIC
        
                idx_val, bt_idx = self.eval(idx_expr)
                try:
                    out = df_obj[idx_val]
                    return out, Lattice.lub(bt_df, bt_idx)
                except Exception:
                    return None, Lattice.DYNAMIC
        
            if tag == "eq":
                _, l, r = node
                lv, bt_l = self.eval(l)
                rv, bt_r = self.eval(r)
                return lv == rv, Lattice.lub(bt_l, bt_r)
        
            if tag == "number":
                return node[1], Lattice.STATIC
        
    
            if tag == "str_get":
                _, recv_ast, idx_ast = node
                recv_val, recv_bt = self.eval(recv_ast)
                idx_val, idx_bt = self.eval(idx_ast)
                try:
                    out = recv_val.str.__getitem__(idx_val)
                    return out, Lattice.lub(recv_bt, idx_bt)
                except Exception:
                    return None, Lattice.DYNAMIC
        
            if tag == "string":
                val = node[1]
                val = getattr(val, "value", val)  
                return val, Lattice.STATIC
               
            if tag == "comparison":
                parts = list(node[1:])
        
                if len(parts) == 3:
                    left, op, right = parts
                    op = getattr(op, "value", op)  
                    return self.eval(("compare", left, op, right))
        

                if len(parts) == 2:
                    left, right = parts
                    return self.eval(("compare", left, "in", right))
        

                assert len(parts) >= 5 and len(parts) % 2 == 1, "malformed comparison"
                result_val = True
                result_bt  = Lattice.STATIC
                cur = parts[0]
                i = 1
                while i < len(parts):
                    op = getattr(parts[i], "value", parts[i])  
                    right = parts[i+1]
                    vv, vbt = self.eval(("compare", cur, op, right))
                    result_bt = Lattice.lub(result_bt, vbt)
                    result_val = bool(vv) and result_val
                    cur = right
                    i += 2
                return result_val, result_bt
                
            if tag == "var":
                name = node[1]
                if name == "pd":
                    return pd, Lattice.STATIC
                if name == "None":
                    return None, Lattice.STATIC
                if name == "True":
                    return True, Lattice.STATIC
                if name == "False":
                    return False, Lattice.STATIC
                if name in ("object","int","float","str","bool","dict","list","tuple"):
                    return {
                        "object": object, "int": int, "float": float, "str": str, "bool": bool,
                        "dict": dict, "list": list, "tuple": tuple
                    }[name], Lattice.STATIC
            
                if name in self.env:
                    return self.env[name], self.bt[name]
                return None, Lattice.DYNAMIC
        
            if tag in ("if", "for", "while", "func_def"):
                return None, Lattice.STATIC if tag == "func_def" else Lattice.DYNAMIC
        
            if tag == "func_call":
                _, name_node, arg_nodes = node
                name = name_node[1] if isinstance(name_node, tuple) else name_node
        
                if name in {"str", "int", "float"} and len(arg_nodes) == 1:
                    val, bt = self.eval(arg_nodes[0])
                    if bt == Lattice.STATIC:
                        if isinstance(val, float) and val.is_integer(): return str(int(val)), Lattice.STATIC
                        if name == "str":   return str(val),  Lattice.STATIC
                        if name == "int":   return int(val),  Lattice.STATIC
                        return float(val),  Lattice.STATIC
                    return None, Lattice.DYNAMIC
        
                if name == "range" and len(arg_nodes) == 1:
                    k, bt = self.eval(arg_nodes[0])
                    if bt == Lattice.STATIC:
                        return list(range(int(k))), Lattice.STATIC
                    return None, Lattice.DYNAMIC
                if name == "STATIC" and len(arg_nodes) == 1:
                    v, _ = self.eval(arg_nodes[0])
                    return v, Lattice.STATIC

                if name == "DYNAMIC" and len(arg_nodes) == 1:
                    v, _ = self.eval(arg_nodes[0])
                    return v, Lattice.DYNAMIC
                return None, Lattice.DYNAMIC
        
            if tag == "compare":
                _, left, op, right = node
                lv, bt_l = self.eval(left)
                rv, bt_r = self.eval(right)
                if op == "in":
                    try:
                        res = lv in rv
                    except Exception:
                        return None, Lattice.DYNAMIC
                    return res, Lattice.lub(bt_l, bt_r)
                op_fn = {
                    "==": lambda a,b: a==b, "!=": lambda a,b: a!=b,
                    ">": lambda a,b: a>b,   "<":  lambda a,b: a<b,
                    ">=":lambda a,b: a>=b,  "<=": lambda a,b: a<=b
                }[op]
                return op_fn(lv, rv), Lattice.lub(bt_l, bt_r)

            if tag == "literal_list":
                raw_items = [it for it in node[1] if not isinstance(it, Token)]
                vals, bt = [], Lattice.STATIC
                for it in raw_items:
                    if isinstance(it, (str, int, float, bool)):
                        vals.append(it)
                    else:
                        v, b = self.eval(it)
                        vals.append(v)
                        bt = Lattice.lub(bt, b)
                return vals, bt
        
            if tag == "func_comp":
                return None, Lattice.DYNAMIC
        
            if tag == "cond":
                _, c, t, e = node
                cv, cbt = self.eval(c)
                if cbt == Lattice.STATIC:
                    return self.eval(t if cv else e)
                tv, tbt = self.eval(t)
                ev, ebt = self.eval(e)
                return (tv if cv else ev), Lattice.lub(tbt, ebt)
        
            raise NotImplementedError(node)

def collect_inlineable(ast_nodes):
    inlineable = {}
    for node in ast_nodes:
        if node[0] != "func_def":
            continue
        _, name_node, params_node, body = node
        fn_name = name_node[1] if isinstance(name_node, tuple) else name_node

        params = params_node if isinstance(params_node, list) else [params_node]
        if len(params) != 1:
            continue
        param_node = params[0]
        if not (isinstance(param_node, tuple) and param_node[0] == "var"):
            continue
        if len(body) != 1 or body[0][0] != "return":
            continue

        param = param_node[1]
        expr = body[0][1]
        inlineable[fn_name] = (param, expr)
    return inlineable
def emit_expr(expr):
    if expr is None: return "None"
    if isinstance(expr, bool): return "True" if expr else "False"
    if isinstance(expr, (int, float)): return repr(expr)
    if isinstance(expr, str):
        s = expr
        if (len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]) or \
           (len(s) >= 2 and s[0] in ('f','F') and s[1] in ("'", '"')):
            return s
        return repr(s)
    if isinstance(expr, list):
        return "[" + ", ".join(emit_expr(e) for e in expr) + "]"

    assert isinstance(expr, tuple) and expr, f"Unknown expr node: {expr!r}"
    tag = expr[0]
    if tag == "cond":
        _, cond_ast, then_ast, else_ast = expr
        return f"{emit_expr(then_ast)} if {emit_expr(cond_ast)} else {emit_expr(else_ast)}"

    if tag == "kwarg":
        _, k, v = expr
        return f"{k}={emit_expr(v)}"
    if tag == "attr_of":
        _, recv, name = expr
        return f"{emit_expr(recv)}.{name}"

    if tag == "loc_lhs":
        _, df_node, rows, cols = expr
        return f"{emit_expr(df_node)}.loc[{emit_expr(rows)}, {emit_expr(cols)}]"
        
    if tag == "literal_dict":
        pairs = []
        for _, k, v in expr[1]:
            pairs.append(f"{emit_expr(k)}: {emit_expr(v)}")
        return "{" + ", ".join(pairs) + "}"
    if tag == "index_get":
        _, base, idx = expr
        if isinstance(base, tuple) and base[0] == "attr_of" and base[2] == "loc":
            if isinstance(idx, tuple) and idx[0] == "literal_list" and len(idx[1]) == 2:
                rows, cols = idx[1]
                return f"{emit_expr(base)}[{emit_expr(rows)}, {emit_expr(cols)}]"
        return f"{emit_expr(base)}[{emit_expr(idx)}]"

    if tag == "comparison":
        _, l, op, r = expr
        op = getattr(op, "value", op)
        return f"{emit_expr(l)} {op} {emit_expr(r)}"
    if tag == "unop":
        _, op, operand = expr
        if op == "-":   return "-" + emit_expr(operand)
        if op == "not": return "not " + emit_expr(operand)

    if tag == "method_call":
        _, receiver, method, args = expr
        recv_code = emit_expr(receiver)
        args_code = ", ".join(emit_expr(a) for a in args)
        return f"{recv_code}.{method}({args_code})"

    if tag == "binop":
        _, l, op, r = expr
        if op == "+" and l[0] == "string" and r[0] == "string":
            return f'"{l[1] + r[1]}"'
        return f"{emit_expr(l)} {op} {emit_expr(r)}"

    if tag == "compare":
        _, l, op, r = expr
        return f"{emit_expr(l)} {op} {emit_expr(r)}"

    if tag == "index":  
        _, base, parts = expr
        return _emit_index(base, parts, emit_expr)
    
    if tag == "filter":
        _, df_node, cond = expr
        df_name = emit_expr(df_node)
        return f"{df_name}[{emit_expr(cond)}]"

    if tag == "col_access":
        _, var, idx = expr
        return f"{emit_expr(var)}[{emit_expr(idx)}]"


    if tag == "literal_tuple":
        inner = ", ".join(emit_expr(it) for it in expr[1])
        return f"({inner})"
    
    if tag == "func_call":
        _, name_node, args = expr
        name = name_node[1] if isinstance(name_node, tuple) and name_node[0] == "var" else str(name_node)
        if name == "DYNAMIC" and len(args) == 1:
            return emit_expr(args[0])
        if name == "STATIC" and len(args) == 1:
            return emit_expr(args[0])
        return f"{name}({', '.join(emit_expr(a) for a in args)})"

    if tag == "str_method_call":
        _, recv, method, args = expr
        recv_code = emit_expr(recv)
        args_code = ", ".join(emit_expr(a) for a in args)
        return f"{recv_code}.str.{method}({args_code})"

    if tag == "number":
        n = expr[1]
        return str(int(n)) if n == int(n) else str(n)

    if tag == "string":
        return f'"{expr[1]}"'

    if tag == "literal_list":
        _, items = expr
        elems = ", ".join(emit_expr(it) for it in items)
        return f"[{elems}]"

    if tag == "var":
        return expr[1]

    if tag == "col":
        _, df_var, col_name = expr
        return f'{df_var}["{col_name}"]'

    if tag == "str_get":
        _, recv, idx = expr
        return f"{emit_expr(recv)}.str[{emit_expr(idx)}]"

    if tag == "func_comp":
        _, f1, f2 = expr
        return f"(lambda __v: {emit_expr(f2)}({emit_expr(f1)}(__v)))"

    raise ValueError(expr)


def _lhs_name(n):
    return n[1] if isinstance(n, tuple) and n and n[0] == "var" else emit_expr(n)


def emit_residual(residual, schema, env, inlineable=None, indent=""):
    lines = []
    IND = "    "
    for node in residual:
        tag = node[0]
        
        if tag == "col_op":
            _, var_node, col_expr, rhs = node
            var_name = _lhs_name(var_node)
            df_obj = env.get(var_name)
            target = (f'{var_name}.loc[:, {emit_expr(col_expr)}]'
                      if (df_obj is not None and id(df_obj) in schema)
                      else f'{var_name}[{emit_expr(col_expr)}]')
            lines.append(f'{indent}{target} = {emit_expr(rhs)}')

        elif tag == "assign":
            _, var_node, rhs = node
            lhs = _lhs_name(var_node)
            rhs_code = emit_expr(rhs) if rhs[0] != "read_csv" else f'pd.read_csv("{rhs[1]}")'
            lines.append(f"{indent}{lhs} = {rhs_code}")
            
        elif tag == "loc_set":
            _, df_node, rows, cols, rhs = node
            df_name = _lhs_name(df_node)
            lhs_str = f'{df_name}.loc[{emit_expr(rows)}, {emit_expr(cols)}]'
        
            def _same_lhs_as_index_get(e):
                return (isinstance(e, tuple) and e[0] == "index_get" and
                        isinstance(e[1], tuple) and e[1][0] == "attr_of" and e[1][1] == df_node and e[1][2] == "loc" and
                        isinstance(e[2], tuple) and e[2][0] == "literal_list" and
                        len(e[2][1]) == 2 and e[2][1][0] == rows and e[2][1][1] == cols)
        
            if isinstance(rhs, tuple) and rhs and rhs[0] == "binop":
                _, lnode, op, rnode = rhs
                if op in {"+", "-", "*", "/"} and _same_lhs_as_index_get(lnode):
                    lines.append(f'{indent}{lhs_str} {op}= {emit_expr(rnode)}')
                    continue
        
            lines.append(f'{indent}{lhs_str} = {emit_expr(rhs)}')
            
        elif tag == "if":
            _, cond, then_blk, else_blk = node
            lines.append(f"{indent}if {emit_expr(cond)}:")
            lines.extend(emit_residual(then_blk, schema, env, inlineable, indent + IND))
            if else_blk:
                lines.append(f"{indent}else:")
                lines.extend(emit_residual(else_blk, schema, env, inlineable, indent + IND))

        elif tag == "for":
            _, var_node, iterable, body = node
            lines.append(f"{indent}for {emit_expr(var_node)} in {emit_expr(iterable)}:")
            lines.extend(emit_residual(body, schema, env, inlineable, indent + IND))

        elif tag == "while":
            _, cond, body = node
            lines.append(f"{indent}while {emit_expr(cond)}:")
            lines.extend(emit_residual(body, schema, env, inlineable, indent + IND))

        elif tag == "import_plain":
            _, module, alias = node
            lines.append(f"{indent}import {module}" + (f" as {alias}" if alias else ""))

        elif tag == "import_from":
            _, module, name, alias = node
            lines.append(f"{indent}from {module} import {name}" + (f" as {alias}" if alias else ""))

        elif tag == "pass_stmt":
            lines.append(f"{indent}pass")

        elif tag == "aug_assign":
            _, lhs, op, rhs = node
            lines.append(f"{indent}{emit_expr(lhs)} {op} {emit_expr(rhs)}")
        
        elif tag == "loc_set":
            _, df_node, rows, cols, rhs = node
            df_name = _lhs_name(df_node) 
            lines.append(f'{indent}{df_name}.loc[{emit_expr(rows)}, {emit_expr(cols)}] = {emit_expr(rhs)}')
            
        elif tag == "func_call":
            _, name_node, args = node
            name = name_node[1] if isinstance(name_node, tuple) else name_node
            lines.append(f"{indent}{name}({', '.join(emit_expr(a) for a in args)})")

    return lines
def _clean_index_parts(parts):
    if not isinstance(parts, list):
        return [parts]
    cleaned = []
    for p in parts:
        if isinstance(p, Token) and p.type in ("LSQB", "RSQB", "COMMA"):
            continue
        cleaned.append(p)
    return cleaned

def _emit_index(base, parts, emit_expr):
    elems = _clean_index_parts(parts)
    if len(elems) == 0:
        inside = ""
    elif len(elems) == 1:
        inside = emit_expr(elems[0])
    else:
        inside = ", ".join(emit_expr(e) for e in elems)
    return f"{emit_expr(base)}[{inside}]"

def _deep_normalize(n):
    if isinstance(n, tuple):
        return tuple(_deep_normalize(c) for c in n)
    if isinstance(n, list):
        return [_deep_normalize(c) for c in n]
    if hasattr(n, "data") and hasattr(n, "children"):
        tag = str(n.data)
        kids = [_deep_normalize(c) for c in n.children]
        if tag == "start":
            flat = []
            for k in kids:
                if isinstance(k, list):
                    flat.extend(k)
                else:
                    flat.append(k)
            return flat
        return tuple([tag] + kids)
    return n

def normalize_ast_list(ast):
    ast = _deep_normalize(ast)
    return ast if isinstance(ast, list) else [ast]

def _assert_no_trees(n, path=()):
    if hasattr(n, "data") and hasattr(n, "children"):
        raise TypeError(f"Residual Tree at {path} with rule {n.data}")
    if isinstance(n, (list, tuple)):
        for i, c in enumerate(n):
            _assert_no_trees(c, path + (i,))

import re

UNSUPPORTED_STMT_RE = re.compile(r'^\s*(class|with|try|except|finally)\b')

def prefilter_for_lark(src: str) -> str:
    out = []
    suppress_stack = []  

    def current_suppressed(indent: int) -> bool:
        return any(indent > base for base in suppress_stack)

    lines = src.splitlines(keepends=True)
    for line in lines:
        raw = line
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)

        while suppress_stack and indent <= suppress_stack[-1]:
            suppress_stack.pop()

        if stripped.startswith("@"): 
            out.append("# NONP-FILTER: " + raw)
            continue

        if current_suppressed(indent):
            out.append("# NONP-FILTER: " + raw)
            continue

        if UNSUPPORTED_STMT_RE.match(raw):
            out.append("# NONP-FILTER: " + raw)
            if raw.rstrip().endswith(":"):
                suppress_stack.append(indent)
            continue

        out.append(raw)

    return "".join(out)

code_for_parser = prefilter_for_lark(code)
tree = parser.parse(code_for_parser)        
        
ast_nodes = parse_code(code_for_parser)
ast_nodes = normalize_ast_list(ast_nodes)


ast_nodes = unroll_loops(ast_nodes, max_unroll=8)
inlineable = collect_inlineable(ast_nodes)
interp = Interpreter(static_files={"m3d1.csv", "m3d2.csv"}, inlineable=inlineable)
interp.run(ast_nodes)

prelude = []
for (_, name_node, params, body) in interp.spec_defs:
    fn = name_node[1] if isinstance(name_node, tuple) else name_node
    ps = ", ".join(p[1] for p in params if isinstance(p, tuple) and p and p[0] == "var")
    prelude.append(f"def {fn}({ps}):")
    body_lines = emit_residual(body, schema=None, env=interp.env, inlineable=inlineable, indent="    ")
    if not any(l.strip().startswith("return ") for l in body_lines):
        body_lines.append("    return None")
    prelude.extend(body_lines)

code_snippet = emit_residual(interp.residual, schema=interp.schema, env=interp.env, inlineable=inlineable)
print("\n".join(prelude + code_snippet))


