import itertools
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form

import dataset as data


# ============================================================
# ============================================================

@dataclass
class ParsedCode:
    label: str
    pCode: list
    signs: list
    comp_lengths: list


def parse_peer_code(code):
    m = re.match(r"\[(.*?)\]\s*/\s*(.*)", code.strip())
    if not m:
        raise ValueError("Code must have the form [ ... ] / signs")

    inside = m.group(1).strip()
    signs = m.group(2).split()

    raw_components = [c.strip() for c in inside.split(",")]
    pCode = []
    comp_lengths = []

    for comp in raw_components:
        if comp == "":
            comp_lengths.append(0)
            continue
        entries = [int(x) for x in comp.split()]
        pCode.extend(entries)
        comp_lengths.append(len(entries))

    if len(signs) != len(pCode):
        raise ValueError(
            f"Number of crossing labels ({len(signs)}) does not match "
            f"number of peer entries ({len(pCode)})."
        )

    return pCode, signs, comp_lengths


def build_successor_map(comp_lengths):
    succ = {}
    start = 0

    for r in comp_lengths:
        labels = list(range(start, start + 2 * r))
        if labels:
            for i in range(len(labels)):
                succ[labels[i]] = labels[(i + 1) % len(labels)]
        start += 2 * r

    return succ


def interval_key(I):
    death = I["death"]
    if death == np.inf:
        death = "inf"
    return (I["dim"], I["birth"], death)


# ============================================================
# Coloring and Biquandle Routines
# ============================================================

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def is_biquandle(under_mat, over_mat):
    over_mat = np.array(over_mat)
    under_mat = np.array(under_mat)
    n = over_mat.shape[0]

    # B1'
    for x in range(n):
        if over_mat[x, x] != under_mat[x, x]:
            return False

    # B2
    seen1 = set()
    for a, b in itertools.product(range(n), repeat=2):
        s = (over_mat[b, a], under_mat[a, b])
        if s in seen1:
            return False
        seen1.add(s)

    for b in range(n):
        seen2 = set()
        for a in range(n):
            s = over_mat[a, b]
            if s in seen2:
                return False
            seen2.add(s)

    for b in range(n):
        seen3 = set()
        for a in range(n):
            s = under_mat[a, b]
            if s in seen3:
                return False
            seen3.add(s)

    # B3
    for x, y, z in itertools.product(range(n), repeat=3):
        left1 = under_mat[under_mat[x, y], under_mat[z, y]]
        right1 = under_mat[under_mat[x, z], over_mat[y, z]]
        if left1 != right1:
            return False

        left2 = over_mat[under_mat[x, y], under_mat[z, y]]
        right2 = under_mat[over_mat[x, z], over_mat[y, z]]
        if left2 != right2:
            return False

        left3 = over_mat[over_mat[x, y], over_mat[z, y]]
        right3 = over_mat[over_mat[x, z], under_mat[y, z]]
        if left3 != right3:
            return False

    return True


def is_endomorphism(f, over_mat, under_mat):
    p = len(f)
    for x in range(p):
        for y in range(p):
            left1 = f[over_mat[x][y]]
            right1 = over_mat[f[x]][f[y]]
            if left1 != right1:
                return False

            left2 = f[under_mat[x][y]]
            right2 = under_mat[f[x]][f[y]]
            if left2 != right2:
                return False

    return True


def find_bq_ends(biquandle):
    p = data.bq_list[biquandle].p
    all_funcs = itertools.product(range(p), repeat=p)
    endos = [
        list(f)
        for f in all_funcs
        if is_endomorphism(
            f,
            data.bq_list[biquandle].over,
            data.bq_list[biquandle].under,
        )
    ]
    return endos


def equationDeriver_link(parsed_code, remove_duplicates=True):
    pCode = parsed_code.pCode
    signs = parsed_code.signs
    comp_lengths = parsed_code.comp_lengths

    succ = build_successor_map(comp_lengths)

    equations = []
    redundantSemiarcs = []

    c = len(pCode)

    for k in range(c):
        yk = abs(pCode[k])

        if pCode[k] > 0 and signs[k] == "+":
            equations.append(("over", 2 * k + 1, yk, 2 * k))
            equations.append(("under", yk, 2 * k + 1, succ[yk]))

        elif pCode[k] > 0 and signs[k] == "-":
            equations.append(("over", yk, 2 * k + 1, succ[yk]))
            equations.append(("under", 2 * k + 1, yk, 2 * k))

        elif pCode[k] < 0 and signs[k] == "+":
            equations.append(("over", 2 * k, succ[yk], 2 * k + 1))
            equations.append(("under", succ[yk], 2 * k, yk))

        elif pCode[k] < 0 and signs[k] == "-":
            equations.append(("over", succ[yk], 2 * k, yk))
            equations.append(("under", 2 * k, succ[yk], 2 * k + 1))

        elif signs[k] == "*":
            equations.append(("virtual", 2 * k, 2 * k + 1, 1))
            equations.append(("virtual", yk, succ[yk], 1))
            redundantSemiarcs.append(succ[yk])
            redundantSemiarcs.append(2 * k + 1)

        else:
            raise ValueError(
                f"Unsupported crossing data at crossing {k}: "
                f"peer entry {pCode[k]}, sign {signs[k]}"
            )

    if remove_duplicates:
        equations = list(dict.fromkeys(equations))
    return equations, redundantSemiarcs, succ


def build_allowed_triples(u_matrix, o_matrix):
    m = len(u_matrix)

    allowed_over = set()
    allowed_under = set()

    for x in range(m):
        for y in range(m):
            allowed_over.add((x, y, o_matrix[x][y]))
            allowed_under.add((x, y, u_matrix[x][y]))

    return {
        "over": allowed_over,
        "under": allowed_under,
    }


def normalize_equations(equations, redundant_semiarcs=None):
    if redundant_semiarcs is None:
        redundant_semiarcs = []

    vars_all = set()
    for eq in equations:
        op, a, b, c = eq
        vars_all.add(a)
        vars_all.add(b)
        if op != "virtual":
            vars_all.add(c)

    uf = UnionFind(vars_all)

    for eq in equations:
        op, a, b, c = eq
        if op == "virtual":
            uf.union(a, b)

    new_equations = []
    for eq in equations:
        op, a, b, c = eq
        if op == "virtual":
            continue
        aa = uf.find(a)
        bb = uf.find(b)
        cc = uf.find(c)
        new_equations.append((op, aa, bb, cc))

    new_equations = list(dict.fromkeys(new_equations))
    canonical_vars = sorted({uf.find(v) for v in vars_all})
    rep_map = {v: uf.find(v) for v in vars_all}

    return new_equations, canonical_vars, rep_map, uf


def revise_constraint(domains, constraint, allowed_triples):
    op, a, b, c = constraint
    triples = allowed_triples[op]

    Da = domains[a]
    Db = domains[b]
    Dc = domains[c]

    good_a = set()
    good_b = set()
    good_c = set()

    for xa, xb, xc in triples:
        if xa in Da and xb in Db and xc in Dc:
            good_a.add(xa)
            good_b.add(xb)
            good_c.add(xc)

    if not good_a or not good_b or not good_c:
        raise ValueError("Constraint contradiction detected.")

    changed = False

    if good_a != Da:
        domains[a] = good_a
        changed = True
    if good_b != Db:
        domains[b] = good_b
        changed = True
    if good_c != Dc:
        domains[c] = good_c
        changed = True

    return changed


def propagate(domains, equations, allowed_triples):
    queue = deque(equations)

    var_to_eqs = defaultdict(list)
    for eq in equations:
        _, a, b, c = eq
        var_to_eqs[a].append(eq)
        var_to_eqs[b].append(eq)
        var_to_eqs[c].append(eq)

    while queue:
        eq = queue.popleft()

        before = {
            eq[1]: domains[eq[1]].copy(),
            eq[2]: domains[eq[2]].copy(),
            eq[3]: domains[eq[3]].copy(),
        }

        changed = revise_constraint(domains, eq, allowed_triples)

        if changed:
            touched = set()
            for v in [eq[1], eq[2], eq[3]]:
                if domains[v] != before[v]:
                    touched.add(v)

            for v in touched:
                for other_eq in var_to_eqs[v]:
                    if other_eq != eq:
                        queue.append(other_eq)

    return domains


def choose_branch_variable(domains):
    candidates = [v for v, D in domains.items() if len(D) > 1]
    if not candidates:
        return None
    return min(candidates, key=lambda v: len(domains[v]))


def solve_domains(domains, equations, allowed_triples, max_solutions=None):
    solutions = []

    try:
        propagate(domains, equations, allowed_triples)
    except ValueError:
        return solutions

    branch_var = choose_branch_variable(domains)

    if branch_var is None:
        solutions.append({v: next(iter(D)) for v, D in domains.items()})
        return solutions

    for value in sorted(domains[branch_var]):
        new_domains = {v: D.copy() for v, D in domains.items()}
        new_domains[branch_var] = {value}

        sub_solutions = solve_domains(
            new_domains,
            equations,
            allowed_triples,
            max_solutions=max_solutions,
        )

        solutions.extend(sub_solutions)

        if max_solutions is not None and len(solutions) >= max_solutions:
            return solutions[:max_solutions]

    return solutions


def expand_solution(solution, rep_map):
    full = {}
    for original_var, rep in rep_map.items():
        full[original_var] = solution[rep]
    return full


def solution_dict_to_tuple(solution):
    n = max(solution.keys()) + 1
    return tuple(solution[i] for i in range(n))


def colorings_of_link(parsed_code, u_matrix, o_matrix, max_solutions=None):
    equations, redundant_semiarcs, succ = equationDeriver_link(parsed_code)

    allowed_triples = build_allowed_triples(u_matrix, o_matrix)

    norm_eqs, canonical_vars, rep_map, uf = normalize_equations(
        equations,
        redundant_semiarcs=redundant_semiarcs,
    )

    m = len(u_matrix)
    domains = {v: set(range(m)) for v in canonical_vars}

    canonical_solutions = solve_domains(
        domains,
        norm_eqs,
        allowed_triples,
        max_solutions=max_solutions,
    )

    canonical_var_order = sorted(canonical_vars)

    canonical_solutions_list = [
        tuple(sol[v] for v in canonical_var_order)
        for sol in canonical_solutions
    ]

    full_solutions_dict = [
        expand_solution(sol, rep_map)
        for sol in canonical_solutions
    ]

    full_solutions = [
        solution_dict_to_tuple(sol)
        for sol in full_solutions_dict
    ]

    return {
        "canonical_equations": norm_eqs,
        "canonical_variables": canonical_var_order,
        "canonical_solutions": canonical_solutions,
        "canonical_solutions_list": canonical_solutions_list,
        "full_solutions": full_solutions,
        "num_colorings": len(full_solutions),
        "rep_map": rep_map,
        "succ": succ,
    }


def coloring_quiver_data(coloring_list, endos):
    M = edge_matrix(coloring_list, endos)
    A = (M > 0).astype(int)

    return {
        "vertices": coloring_list,
        "edge_matrix": M,
        "adjacency_matrix": A,
    }


# ============================================================
# N-Directed Clique Homology
# ============================================================

def edge_matrix(verts, endos):
    edges = np.zeros((len(verts), len(verts)), dtype=int)
    for endo in endos:
        for t in range(len(verts)):
            image = []
            for i in range(len(verts[t])):
                image.append(endo[verts[t][i]])
            for h in range(len(verts)):
                if image == list(verts[h]):
                    edges[t][h] = edges[t][h] + 1
    return edges


def simplicer_fast(edges, N):
    n = len(edges)
    simplicies = defaultdict(list)

    for v in range(n):
        simplicies[0].append([v])

    outN = []
    for u in range(n):
        nbrs = {v for v in range(n) if v != u and edges[u][v] >= N}
        outN.append(nbrs)

    def extend(prefix, candidates):
        for w in list(candidates):
            new_prefix = prefix + [w]
            simplicies[len(new_prefix) - 1].append(new_prefix)

            new_candidates = candidates.intersection(outN[w])
            new_candidates = new_candidates - {w}

            for used in new_prefix:
                new_candidates.discard(used)

            if new_candidates:
                extend(new_prefix, new_candidates)

    for v in range(n):
        extend([v], outN[v].copy())

    return simplicies


def normalize_K(usot_K):
    max_d = max(usot_K.keys()) if usot_K else 0
    K = []
    for d in range(max_d + 1):
        raw = usot_K.get(d, [])
        K.append([tuple(s) for s in raw])
    while len(K) > 0 and len(K[-1]) == 0:
        K.pop()
    return K


def build_boundary(K_prev, K_cur):
    idx_prev = {tau: i for i, tau in enumerate(K_prev)}
    rows, cols = len(K_prev), len(K_cur)
    D = sp.zeros(rows, cols, domain=sp.ZZ)

    for j, sigma in enumerate(K_cur):
        m = len(sigma)
        for k in range(m):
            tau = sigma[:k] + sigma[k + 1:]
            i = idx_prev[tau]
            D[i, j] += 1 if (k % 2 == 0) else -1

    return D


def clear_denominators(M: sp.Matrix) -> sp.Matrix:
    if M.cols == 0:
        return sp.zeros(M.rows, 0, domain=sp.ZZ)

    MZ = sp.zeros(M.rows, M.cols, domain=sp.ZZ)
    for j in range(M.cols):
        col = M[:, j]
        den = 1
        for x in col:
            den = sp.ilcm(den, sp.denom(x))
        MZ[:, j] = (col * den).applyfunc(sp.simplify)

    return sp.Matrix(MZ)


def integer_kernel_basis(D):
    NS = D.nullspace()
    if not NS:
        return sp.zeros(D.cols, 0, domain=sp.ZZ)

    KQ = sp.Matrix.hstack(*NS)
    if KQ.rows != D.cols:
        KQ = KQ.reshape(D.cols, len(NS))

    return clear_denominators(KQ)


def homology_Z(usot_K):
    K = normalize_K(usot_K)
    dims = [len(Kd) for Kd in K]
    max_n = len(K) - 1

    boundaries = [None] * (max_n + 1)
    for n in range(1, max_n + 1):
        boundaries[n] = build_boundary(K[n - 1], K[n])

    ker_basis = [None] * (max_n + 1)
    ker_basis[0] = sp.eye(dims[0])

    for n in range(1, max_n + 1):
        ker_basis[n] = integer_kernel_basis(boundaries[n])

    H = []
    reps = [[] for _ in range(max_n + 1)]

    for n in range(0, max_n + 1):
        Kmat = ker_basis[n]
        k = Kmat.shape[1] if isinstance(Kmat, sp.Matrix) else 0

        if n == max_n:
            H.append({"beta": int(k), "torsion": []})
            for j in range(k):
                col = Kmat[:, j]
                reps[n].append(
                    [(i, int(col[i])) for i in range(len(col)) if col[i] != 0]
                )
            continue

        B = boundaries[n + 1]
        if B is None or B.cols == 0 or k == 0:
            H.append({"beta": int(k), "torsion": []})
            for j in range(k):
                col = Kmat[:, j]
                reps[n].append(
                    [(i, int(col[i])) for i in range(len(col)) if col[i] != 0]
                )
            continue

        G = Kmat.T * Kmat
        YQ = sp.zeros(k, B.cols)
        for j in range(B.cols):
            YQ[:, j] = G.LUsolve(Kmat.T * B[:, j])

        YZ = clear_denominators(YQ)
        DY = smith_normal_form(YZ, domain=sp.ZZ)
        diag = [
            int(abs(DY[i, i]))
            for i in range(min(DY.rows, DY.cols))
            if DY[i, i] != 0
        ]

        rY = len(diag)
        torsion = [d for d in diag if d > 1]
        beta = k - rY
        H.append({"beta": int(beta), "torsion": torsion})

        N_list = (YQ.T).nullspace()
        if beta == 0 or not N_list:
            continue

        NQ = sp.Matrix.hstack(*N_list)
        N_int = clear_denominators(NQ)

        for j in range(N_int.cols):
            col = N_int[:, j]
            g = 0
            for x in col:
                g = int(sp.igcd(g, int(abs(x))))
            if g > 1:
                N_int[:, j] = (col / g).applyfunc(int)

        C = Kmat * N_int
        for j in range(C.cols):
            col = C[:, j]
            reps[n].append(
                [(i, int(col[i])) for i in range(len(col)) if col[i] != 0]
            )

    return dims, boundaries, H, reps


# ============================================================
# Persistence
# ============================================================

class PersistenceOnePass:
    def __init__(self, p: int = 2):
        if p < 2:
            raise ValueError("Field modulus p must be prime and >= 2.")
        self.p = p

    def _add_cols(self, a: dict, b: dict, scale=1):
        p = self.p
        for r, c in b.items():
            z = (a.get(r, 0) + scale * c) % p
            if z == 0:
                a.pop(r, None)
            else:
                a[r] = z

    @staticmethod
    def _leading_row(col: dict):
        return max(col) if col else None

    def _inv(self, x: int) -> int:
        return pow(x, -1, self.p)

    def compute_intervals(self, simplices: List[Dict[str, Any]]):
        T = {}
        marked_zero = [False] * len(simplices)
        intervals = defaultdict(list)

        for j, s in enumerate(simplices):
            d = {}
            for i in s["bd"]:
                d[i] = (d.get(i, 0) + 1) % self.p
                if d[i] == 0:
                    d.pop(i, None)

            i = self._leading_row(d)
            while i is not None and i in T:
                ci = d[i] % self.p
                pi = T[i][i] % self.p
                scale = (ci * self._inv(pi)) % self.p
                self._add_cols(d, T[i], scale=(self.p - scale))
                i = self._leading_row(d)

            if not d:
                marked_zero[j] = True
            else:
                i = self._leading_row(d)
                T[i] = d
                birth_t = simplices[i]["t"]
                death_t = s["t"]
                dim_bar = simplices[i]["dim"]
                intervals[dim_bar].append((birth_t, death_t))

        killed_rows = set(T.keys())
        for j, s in enumerate(simplices):
            if marked_zero[j] and j not in killed_rows:
                intervals[s["dim"]].append((s["t"], None))

        for d in intervals:
            intervals[d].sort(
                key=lambda x: (x[0], float("inf") if x[1] is None else x[1])
            )

        return dict(intervals)


def print_intervals_with_multiplicity(intervals, keep_zero=False):
    counter = Counter()

    for I in intervals:
        if (not keep_zero) and I["death"] != np.inf and I["birth"] == I["death"]:
            continue
        counter[interval_key(I)] += 1

    grouped = defaultdict(list)
    for (dim, birth, death), mult in sorted(
        counter.items(),
        key=lambda x: (
            x[0][0],
            x[0][1],
            x[0][2] if x[0][2] != "inf" else float("inf"),
        ),
    ):
        grouped[dim].append((birth, death, mult))

    for dim in sorted(grouped.keys()):
        print(f"H_{dim}:")
        for birth, death, mult in grouped[dim]:
            death_str = "∞" if death == "inf" else str(death)
            print(f"  [{birth}, {death_str}) x {mult}")


def edge_matrix_filt(verts, S_filt):
    return [edge_matrix(verts, S) for S in S_filt]


def simplicial_filt(edge_filt, N):
    return [simplicer_fast(E, N) for E in edge_filt]


def filtration_check(K_filt):
    for i in range(len(K_filt) - 1):
        Ki = normalize_K(K_filt[i])
        Kj = normalize_K(K_filt[i + 1])

        max_dim = max(len(Ki), len(Kj))
        for d in range(max_dim):
            Ai = set(Ki[d]) if d < len(Ki) else set()
            Aj = set(Kj[d]) if d < len(Kj) else set()

            if not Ai.issubset(Aj):
                return False, i, d, Ai - Aj

    return True, None, None, None


def simplex_births(K_filt):
    births = {}
    for t, K in enumerate(K_filt):
        NK = normalize_K(K)
        for d, simplices in enumerate(NK):
            for s in simplices:
                if s not in births:
                    births[s] = {"dim": d, "birth": t}
    return births


def global_simplices_from_births(births):
    simplices = []
    for s, info in births.items():
        simplices.append((s, info["dim"], info["birth"]))

    simplices.sort(key=lambda x: (x[2], x[1], x[0]))
    return simplices


def simplex_boundary_f2(sigma):
    if len(sigma) == 1:
        return []

    bd = []
    for i in range(len(sigma)):
        face = sigma[:i] + sigma[i + 1:]
        bd.append(face)
    return bd


def build_boundary_matrix_f2(global_simplices):
    simplex_to_col = {s[0]: j for j, s in enumerate(global_simplices)}
    n = len(global_simplices)
    D = np.zeros((n, n), dtype=int)

    for j, (sigma, dim, birth) in enumerate(global_simplices):
        if dim == 0:
            continue

        for face in simplex_boundary_f2(sigma):
            i = simplex_to_col.get(face, None)
            if i is None:
                raise ValueError(
                    f"Face {face} of simplex {sigma} not found in global simplex list."
                )
            D[i, j] = (D[i, j] + 1) % 2

    return D


def reduce_boundary_matrix_f2(D):
    R = D.copy()
    n = R.shape[1]

    low = {}
    low_of_col = {}

    for j in range(n):
        while True:
            rows = np.where(R[:, j] == 1)[0]
            if len(rows) == 0:
                break

            l = rows[-1]
            if l in low:
                R[:, j] = (R[:, j] + R[:, low[l]]) % 2
            else:
                low[l] = j
                low_of_col[j] = l
                break

    return R, low, low_of_col


def extract_intervals_f2(global_simplices, low, low_of_col):
    intervals = []

    paired_birth_cols = set(low.keys())
    paired_death_cols = set(low_of_col.keys())

    for birth_row, death_col in low.items():
        birth_simplex, birth_dim, birth_time = global_simplices[birth_row]
        death_simplex, death_dim, death_time = global_simplices[death_col]

        intervals.append({
            "dim": birth_dim,
            "birth": birth_time,
            "death": death_time,
            "birth_simplex": birth_simplex,
            "death_simplex": death_simplex,
        })

    for j, (sigma, dim, birth) in enumerate(global_simplices):
        col_is_zero = (j not in low_of_col)
        never_killed = (j not in paired_birth_cols)

        if col_is_zero and never_killed:
            intervals.append({
                "dim": dim,
                "birth": birth,
                "death": np.inf,
                "birth_simplex": sigma,
                "death_simplex": None,
            })

    intervals.sort(key=lambda x: (x["dim"], x["birth"], x["death"]))
    return intervals


def persistent_homology_f2(K_filt):
    births = simplex_births(K_filt)
    global_simplices = global_simplices_from_births(births)
    D = build_boundary_matrix_f2(global_simplices)
    R, low, low_of_col = reduce_boundary_matrix_f2(D)
    intervals = extract_intervals_f2(global_simplices, low, low_of_col)
    return global_simplices, D, R, intervals


def deadborn_matrix_from_intervals(intervals, num_stages=None):
    if not intervals:
        if num_stages is None:
            return []
        return []

    max_dim = max(I["dim"] for I in intervals)

    if num_stages is None:
        max_stage = max(I["birth"] for I in intervals)
        num_stages = max_stage + 1

    counter = Counter()

    for I in intervals:
        if I["death"] != np.inf and I["birth"] == I["death"]:
            counter[(I["dim"], I["birth"])] += 1

    M = [[0 for _ in range(num_stages)] for _ in range(max_dim + 1)]

    for (dim, stage), mult in counter.items():
        M[dim][stage] = mult

    return M


def print_deadborn_matrix(M):
    if not M:
        print("Stillborn matrix is empty.")
        return

    num_cols = len(M[0])
    max_entry = max(max(row) for row in M)
    width = max(len(str(max_entry)), 1) + 2

    header = " " * 6 + "".join(f"{j:>{width}}" for j in range(num_cols))
    print(header)
    print(" ")

    for i, row in enumerate(M):
        row_str = "".join(f"{x:>{width}}" for x in row)
        print(f"H_{i}   {row_str}")


def build_endomorphism_filtration(endos, filtration_tokens):
    if filtration_tokens is None:
        raise ValueError("No filtration specification provided.")

    filt = [[]]
    current_indices = []
    current_stage = []
    seen = set()

    for token in filtration_tokens:
        token = token.strip()
        if token == "":
            continue

        ends_stage = token.endswith(",")
        if ends_stage:
            token = token[:-1].strip()

        if token != "":
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(f"Invalid filtration token: '{token}'")

            if not (0 <= idx < len(endos)):
                raise ValueError(
                    f"Endomorphism index out of range: {idx}. "
                    f"Valid range: 0-{len(endos)-1}"
                )

            if idx in seen:
                raise ValueError(
                    f"Endomorphism index {idx} appears more than once in the filtration."
                )

            current_stage.append(idx)
            seen.add(idx)

        if ends_stage:
            current_indices.extend(current_stage)
            filt.append([endos[i] for i in current_indices])
            current_stage = []

    if current_stage:
        current_indices.extend(current_stage)
        filt.append([endos[i] for i in current_indices])

    return filt


def build_endomorphism_filtration_indices(endos, filtration_tokens):
    if filtration_tokens is None:
        raise ValueError("No filtration specification provided.")

    filt = [[]]
    current_indices = []
    current_stage = []
    seen = set()

    for token in filtration_tokens:
        token = token.strip()
        if token == "":
            continue

        ends_stage = token.endswith(",")
        if ends_stage:
            token = token[:-1].strip()

        if token != "":
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(f"Invalid filtration token: '{token}'")

            if not (0 <= idx < len(endos)):
                raise ValueError(
                    f"Endomorphism index out of range: {idx}. "
                    f"Valid range: 0-{len(endos)-1}"
                )

            if idx in seen:
                raise ValueError(
                    f"Endomorphism index {idx} appears more than once in the filtration."
                )

            current_stage.append(idx)
            seen.add(idx)

        if ends_stage:
            if not current_stage:
                raise ValueError("Empty filtration stage is not allowed.")
            current_indices.extend(current_stage)
            filt.append(current_indices.copy())
            current_stage = []

    if current_stage:
        current_indices.extend(current_stage)
        filt.append(current_indices.copy())

    return filt


# ============================================================
# 
# ============================================================

def inv(n, p):
    for i in range(p):
        if (i * n) % p == 1:
            return i


def matrix_converter(matrix):
    n = len(matrix)
    new_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                new_matrix[n - 1][n - 1] = matrix[0][0]
            if i == 0 and j != 0:
                new_matrix[n - 1][j - 1] = matrix[0][j]
            if i != 0 and j == 0:
                new_matrix[i - 1][n - 1] = matrix[i][0]
            if i != 0 and j != 0:
                new_matrix[i - 1][j - 1] = matrix[i][j]
    return new_matrix


def alexander_bq(n, t, r):
    alex_over = np.zeros((n, n), dtype=int)
    alex_under = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            alex_under[i][j] = (t * i + (r - t) * j) % n
            alex_over[i][j] = (r * i) % n

    return alex_under, alex_over


def find_alex_bq_ends(p, t, r):
    all_funcs = itertools.product(range(p), repeat=p)
    under, over = alexander_bq(p, t, r)
    endos = [list(f) for f in all_funcs if is_endomorphism(f, over, under)]
    return endos


def alex_coloring_calc(code, p, t, r):
    pCode = code.pCode
    signs = code.signs
    n = 2 * len(pCode)
    matrix = np.zeros((n, n))

    for i in range(int(n / 2)):
        if pCode[i] > 0 and signs[i] == "+":
            matrix[2 * i, 2 * i] = -1
            matrix[2 * i, 2 * i + 1] = r
            matrix[2 * i + 1, abs(pCode[i])] = t
            matrix[2 * i + 1, 2 * i + 1] = r - t
            matrix[2 * i + 1, (abs(pCode[i]) + 1) % n] = -1

        elif pCode[i] > 0 and signs[i] == "-":
            matrix[2 * i, (abs(pCode[i]) + 1) % n] = -1
            matrix[2 * i, abs(pCode[i])] = r
            matrix[2 * i + 1, 2 * i + 1] = t
            matrix[2 * i + 1, abs(pCode[i])] = r - t
            matrix[2 * i + 1, 2 * i] = -1

        elif pCode[i] < 0 and signs[i] == "+":
            matrix[2 * i, 2 * i + 1] = -1
            matrix[2 * i, 2 * i] = r
            matrix[2 * i + 1, (abs(pCode[i]) + 1) % n] = t
            matrix[2 * i + 1, 2 * i] = r - t
            matrix[2 * i + 1, abs(pCode[i])] = -1

        elif pCode[i] < 0 and signs[i] == "-":
            matrix[2 * i, abs(pCode[i])] = -1
            matrix[2 * i, (abs(pCode[i]) + 1) % n] = r
            matrix[2 * i + 1, 2 * i] = t
            matrix[2 * i + 1, (abs(pCode[i]) + 1) % n] = r - t
            matrix[2 * i + 1, 2 * i + 1] = -1

    matrix = matrix.astype(int)

    basis = kernel_mod(matrix, p)
    M = ensure_array(basis)
    return span_over_Zp(M, p)