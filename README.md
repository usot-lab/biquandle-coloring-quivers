# Biquandle Coloring Quivers and Persistent Homology

---

## Requirements

- Python 3
- SymPy
- NumPy

(Developed using Anaconda.)

---

## Overview

This code computes invariants of links arising from:
- biquandles
- biquandle coloring quivers,
- N-directed clique homology,
- and their associated persistence structures.

These include:
- counting invariants,
- quiver edge matrices,
- N-directed homology groups,
- persistence barcodes,
- and stillborn matrices.

---

# How to run the code

1. Clone the repository:
   ```bash
   git clone https://github.com/usot-lab/biquandle-coloring-quivers.git
   cd biquandle-coloring-quivers
   ```
2. Install dependencies:
   ```bash
   pip install sympy numpy
   ```
3. Run a computation:
   ```bash
   python main.py count --list
   ```
---

## Quick Start

For example, to compute a counting invariant:

```bash
python main.py count --code 2.3.1 --biquandle 1

---

## Files

- `main.py` – command-line interface for running computations
- `functions.py` – core computational routines
- `dataset.py` – link codes, biquandles used in the paper

---

## Usage

List available link codes (labels) and biquandles (indices) in the dataset:
```bash
python main.py count --list
```

Validate a biquandle in the dataset or check if a new data you add to the biquandle storage actually is a valid biquandle:
```bash
python main.py validate --biquandle <INDEX>
```
Example :
```bash
python main.py validate --biquandle 1
```

Compute the biquandle counting invariant of a link with the given biquandle:
```bash
python main.py count --code <LABEL> --biquandle <INDEX>
```
Example :
```bash
python main.py count --code 2.3.1 --biquandle 1
```

List the biquandle endomorphisms of a given biquandle :
```bash
python main.py bq_quiver --biquandle <INDEX> --show-endos
```
Example :
```bash
python main.py bq_quiver --biquandle 1 --show-endos
```

Compute the edge matrix of the biquandle coloring quiver of a link with the given biquandle and the given subset S of endomorphisms:
```bash
python main.py bq_quiver --code <LABEL> --biquandle <INDEX> --endos <INDEX LIST>
```
Example :
```bash
python main.py bq_quiver --code 2.3.1 --biquandle 1 --endos 1 2 3 4
```

Compute the n-th N-directed homology groups of a link with the given biquandle, the given subset S of endomorphisms and the given parameter N:
```bash
python main.py ndch --code <LABEL> --biquandle <INDEX> --endos <INDEX LIST> --N <POSITIVE INTEGER>
```
Example :
```bash
python main.py ndch --code 2.3.1 --biquandle 1 --endos 1 2 3 4 --N 1
```

Compute the persistence barcode and the stillborn matrix of a link with the given biquandle, the given filtration S_* of endomorphisms and the given parameter N:
```bash
python main.py pndch --code <INDEX> --biquandle <INDEX> --filtration <INDEX LIST> --N <POSITIVE INTEGER>
```
Note that the filtration is specified by grouping indices with commas.
Each group represents the new endomorphisms added at that stage.

Example :
```bash
python main.py pndch --code 2.3.1 --biquandle 1 --filtration 1, 2, 3 4, 5 --N 1
```

Examples in the paper :

Example 8
```bash
python main.py ndch --code 3.5.3 --biquandle 13 --endos 3 5 7 9 11 13 15 --N 2
```
```bash
python main.py ndch --code 2.4.9 --biquandle 13 --endos 3 5 7 9 11 13 15 --N 2
```

Example 9
```bash
python main.py bq_quiver --biquandle 7 --show-endos
```
```bash
python main.py bq_filt --code 2.1.1 --biquandle 7 --filtration 1, 2, 3, 4, 0
```

Example 11
```bash
python main.py count --code 2.3.9 --biquandle 20
```
```bash
python main.py count --code 2.4.3 --biquandle 20
```
```bash
python main.py bq_quiver --biquandle 20 --show-endos
```
```bash
python main.py pndch --code 2.3.9 --biquandle 20 --filtration 0 1, 7 13, 9 15 17 19 --N 2
```
```bash
python main.py pndch --code 2.4.3 --biquandle 20 --filtration 0 1, 7 13, 9 15 17 19 --N 2
```

Example 13
```bash
python main.py bq_quiver --biquandle 17 --show-endos
```
```bash
python main.py pndch --code 2.6.0 --biquandle 17 --filtration 0, 1, 5 45 --N 1
```
```bash
python main.py pndch --code 2.6.1 --biquandle 17 --filtration 0, 1, 5 45 --N 1
```

Table 3
```bash
python main.py bq_quiver --biquandle 1 --show-endos
```
```bash
python main.py pndch --code 2.2.1 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 2.3.1 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 2.3.2 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 2.3.6 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 2.3.9 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 2.3.10 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 3.3.0 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 2.6.0 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 3.6.0 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```
```bash
python main.py pndch --code 1.8.0 --biquandle 1 --filtration 1, 2, 4, 5 --N 1
```

Notes :

Biquandles are indexed by integers
corresponding to their position in the `dataset.py`.



