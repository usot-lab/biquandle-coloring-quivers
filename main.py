# main.py
import argparse
from sympy import pprint
import dataset as data
import functions as hp


def print_available_links():
    print("Available link labeled peer codes:")
    print("  " + ", ".join(sorted(c.label for c in data.peers_V)))


def print_available_biquandles():
    print("Available biquandles:")
    for i, bq in enumerate(data.bq_list):
        print(f"{i}: Z_{bq.p}")


def get_code_by_label(code_label):
    for code in data.peers_V:
        if str(code.label) == str(code_label):
            return code
    raise SystemExit(f"Unknown code label: {code_label}")


def check_biquandle_index(index):
    if index is None:
        raise SystemExit("Biquandle index is required.")
    if not (0 <= index < len(data.bq_list)):
        raise SystemExit(
            f"Biquandle index out of range: {index}. "
            f"Valid range: 0-{len(data.bq_list)-1}"
        )

def run_show_bq(args):
    if args.biquandle is None:
        raise SystemExit("Usage: python main.py show_bq --biquandle <INDEX>")

    check_biquandle_index(args.biquandle)
    bq = data.bq_list[args.biquandle]

    print(f"Biquandle X_{args.biquandle} over Z_{bq.p}\n")

    print("Under operation:")
    for row in bq.under:
        print(row)

    print("\nOver operation:")
    for row in bq.over:
        print(row)

def run_bq_quiver(args):
    if args.biquandle is None:
        raise SystemExit("Usage: python main.py bq_quiver --biquandle <INDEX> [--show-endos]")

    check_biquandle_index(args.biquandle)
    bq = data.bq_list[args.biquandle]

    endos = hp.find_bq_ends(args.biquandle)

    if args.show_endos:
        print(f"Endomorphisms of biquandle {args.biquandle}:")
        for i, endo in enumerate(endos):
            print(f"{i}: {endo}")
        return

    if args.code is None or args.endos is None:
        raise SystemExit(
            "Usage: python main.py bq_quiver --code <LABEL> --biquandle <INDEX> --endos <i1> <i2> ..."
        )

    code = get_code_by_label(args.code)

    result = hp.colorings_of_link(code, bq.under, bq.over)
    coloring_list = result["full_solutions"]
    coloring_list_r = result["canonical_solutions_list"]

    chosen_endos = []
    for idx in args.endos:
        if not (0 <= idx < len(endos)):
            raise SystemExit(
                f"Endomorphism index out of range: {idx}. "
                f"Valid range: 0-{len(endos)-1}"
            )
        chosen_endos.append(endos[idx])

    quiver = hp.coloring_quiver_data(coloring_list_r, chosen_endos)

    print("\nEdge matrix:")
    print(quiver["edge_matrix"])

    print("\nVertices:")
    for i, v in enumerate(quiver["vertices"]):
        print(f"{i}: {v}")

def run_count(args):
    if args.code is None or args.biquandle is None:
        raise SystemExit("Usage: python main.py count --code <LABEL> --biquandle <INDEX>")

    check_biquandle_index(args.biquandle)
    code = get_code_by_label(args.code)
    bq = data.bq_list[args.biquandle]

    result = hp.colorings_of_link(code, bq.under, bq.over)
    coloring_list = result["full_solutions"]
    coloring_list_reduced = result["canonical_solutions_list"]

    print(f"The number of colorings of {code.label} with biquandle {args.biquandle} is: {len(coloring_list)}")

    if coloring_list_reduced:
        print(f"The colorings of {code.label} with biquandle {args.biquandle} are:")
        for coloring in coloring_list_reduced:
            print(coloring)

def run_edge_matrix(args):
    if args.code is None or args.biquandle is None or args.filtration is None:
        raise SystemExit(
            "Usage: python main.py bq_filt --code <LABEL> --biquandle <INDEX> --filtration <stage1>, <stage2>"
        )

    check_biquandle_index(args.biquandle)
    code = get_code_by_label(args.code)
    bq = data.bq_list[args.biquandle]

    result = hp.colorings_of_link(code, bq.under, bq.over)
    coloring_list = result["full_solutions"]
    coloring_list_r = result["canonical_solutions_list"]

    endos = hp.find_bq_ends(args.biquandle)
    verts = coloring_list
    S_filt = hp.build_endomorphism_filtration(endos, args.filtration)
    mats = hp.edge_matrix_filt(verts, S_filt)

    print("\nVertices:")
    for i, v in enumerate(result["canonical_solutions_list"]):
        print(f"{i}: {v}")

    print("Edge matrices of the biquandle coloring quiver filtration:\n")
    for i, (S, M) in enumerate(zip(S_filt, mats)):
        print(f"S_{i} = {S}")
        print(M)
        print()


def run_validate(args):
    if args.biquandle is None:
        raise SystemExit("Usage: python main.py validate --biquandle <INDEX>")

    check_biquandle_index(args.biquandle)
    bq = data.bq_list[args.biquandle]

    if hp.is_biquandle(bq.under, bq.over):
        print("These under/over operation matrices")
        pprint(bq.under)
        pprint(bq.over)
        print(f"define a biquandle structure over Z_{bq.p}")
    else:
        print(f"Biquandle X_{args.biquandle} is not a valid biquandle.")


def print_homology(H):
    print("\n=== Homology Groups ===")
    for n, info in enumerate(H):
        beta = info["beta"]
        torsion = info["torsion"]
        if torsion:
            torsion_str = " ⊕ ".join([f"Z/{t}Z" for t in torsion])
            print(f"H_{n} ≅ Z^{beta} ⊕ {torsion_str}")
        else:
            print(f"H_{n} ≅ Z^{beta}")


def run_ndch(args):
    if args.biquandle is None:
        raise SystemExit("Usage: python main.py ndch --biquandle <INDEX> [--show-endos] "
                         "or python main.py ndch --code <LABEL> --biquandle <INDEX> --endos ... --N <N>")

    check_biquandle_index(args.biquandle)
    bq = data.bq_list[args.biquandle]

    endos = hp.find_bq_ends(args.biquandle)

    if args.show_endos:
        print(f"Endomorphisms of biquandle {args.biquandle}:")
        for i, endo in enumerate(endos):
            print(f"{i}: {endo}")

    if args.code is None and args.show_endos and args.endos is None and args.N is None:
        return

    if args.code is None or args.endos is None or args.N is None:
        raise SystemExit(
            "Usage: python main.py ndch --code <LABEL> --biquandle <INDEX> --endos <i1> <i2> ... --N <N>"
        )

    if args.N < 1:
        raise SystemExit("N must be a positive integer.")

    code = get_code_by_label(args.code)

    result = hp.colorings_of_link(code, bq.under, bq.over)
    coloring_list = result["full_solutions"]
    coloring_list_reduced = result["canonical_solutions_list"]

    print(f"Number of colorings of {code.label} with {args.biquandle}: {result['num_colorings']}")
    for sol in coloring_list_reduced:
        print(sol)

    chosen_endos = []
    for idx in args.endos:
        if not (0 <= idx < len(endos)):
            raise SystemExit(
                f"Endomorphism index out of range: {idx}. "
                f"Valid range: 0-{len(endos)-1}"
            )
        chosen_endos.append(endos[idx])

    print("\nChosen endomorphism indices:")
    print(args.endos)

    print("\nChosen endomorphisms:")
    for idx in args.endos:
        print(f"{idx}: {endos[idx]}")

    edge_matrix = hp.edge_matrix(coloring_list, chosen_endos)
    simplices = hp.simplicer_fast(edge_matrix, args.N)
    dims, boundaries, H, reps = hp.homology_Z(simplices)

    print(f"\nN = {args.N}")
    print_homology(H)

def run_pndch(args):
    if args.code is None or args.biquandle is None or args.filtration is None or args.N is None:
        raise SystemExit(
            "Usage: python main.py pndch --code <LABEL> --biquandle <INDEX> --filtration <stage1> <stage2> ... --N <N>"
        )

    check_biquandle_index(args.biquandle)

    if args.N < 1:
        raise SystemExit("N must be a positive integer.")

    code = get_code_by_label(args.code)
    bq = data.bq_list[args.biquandle]

    result = hp.colorings_of_link(code, bq.under, bq.over)
    coloring_list = result["full_solutions"]

    print(f"Number of colorings of {code.label} with X_{args.biquandle}: {result['num_colorings']}")

    endos = hp.find_bq_ends(args.biquandle)

    if args.show_endos:
        print(f"Endomorphisms of biquandle {args.biquandle}:")
        for i, endo in enumerate(endos):
            print(f"{i}: {endo}")
    try:
        S_filt = hp.build_endomorphism_filtration(endos, args.filtration)
        S_filt_indices = hp.build_endomorphism_filtration_indices(endos, args.filtration)
    except ValueError as e:
        raise SystemExit(str(e))

    print("\nFiltration of endomorphism indices:")
    for stage, S in enumerate(S_filt_indices):
        print(f"S_{stage}: {S}")

    E_filt = hp.edge_matrix_filt(coloring_list, S_filt)
    K_filt = hp.simplicial_filt(E_filt, args.N)

    #print("\nLengths:")
    #print("len(S_filt) =", len(S_filt))
    #print("len(E_filt) =", len(E_filt))
    #print("len(K_filt) =", len(K_filt))

    ok, i, d, missing = hp.filtration_check(K_filt)
    if ok:
        print("\nThe filtration is valid.")
    else:
        print("\nThe filtration is NOT valid.")
        print(f"Failure at stage {i}, dimension {d}.")
        print("Missing simplices:")
        for s in missing:
            print(s)
        return

    global_simplices, D, R, intervals = hp.persistent_homology_f2(K_filt)

    #print(f"\nPersistent N-directed clique homology for N = {args.N}:")
    print("\nPersistence intervals with multiplicities:")
    hp.print_intervals_with_multiplicity(intervals, keep_zero=False)
    
    print("\nStillborn matrix:")
    M = hp.deadborn_matrix_from_intervals(intervals, num_stages=len(K_filt))
    hp.print_deadborn_matrix(M)

def main():
    parser = argparse.ArgumentParser(
        description="Computations for link colorings, biquandles, and N-directed clique homology."
    )

    parser.add_argument(
        "action",
        choices=["count", "validate", "show_bq", "bq_quiver", "bq_filt", "ndch", "pndch"],
        help="Action to perform."
    )

    parser.add_argument(
        "--code",
        required=False,
        help="Link code label (e.g. 3.1.10 or 18, depending on dataset labels)."
    )

    parser.add_argument(
        "--biquandle",
        required=False,
        type=int,
        help="Biquandle index."
    )

    parser.add_argument(
        "--filtration",
        nargs="+",
        help="Filtration stages written cumulatively by additions, e.g. --filtration 1 2 4,5"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available link labels and biquandle indices, then exit."
    )

    parser.add_argument(
        "--show-endos",
        action="store_true",
        help="Print the endomorphism list for the selected biquandle."
    )

    parser.add_argument(
        "--endos",
        nargs="+",
        type=int,
        help="Indices of chosen endomorphisms, e.g. --endos 0 13 3 43 45"
    )

    parser.add_argument(
        "--N",
        type=int,
        help="Threshold N for the N-directed clique complex."
    )

    args = parser.parse_args()

    if args.list:
        print_available_links()
        print()
        print_available_biquandles()
        return

    if args.action == "count":
        run_count(args)
    elif args.action == "validate":
        run_validate(args)
    elif args.action == "bq_quiver":
        run_bq_quiver(args)
    elif args.action == "ndch":
        run_ndch(args)
    elif args.action == "pndch":
        run_pndch(args)
    elif args.action == "show_bq":
        run_show_bq(args)
    elif args.action == "bq_filt":
        run_edge_matrix(args)


if __name__ == "__main__":
    main()