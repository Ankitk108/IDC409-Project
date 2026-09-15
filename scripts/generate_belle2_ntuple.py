#!/usr/bin/env python3
"""Generate one labeled continuum-suppression ntuple from a Belle II MDST.

Run this script only inside a configured Belle II basf2 environment. The generated
ROOT file is private input data and must be kept outside Git. Process signal and
background inputs in separate invocations so their provenance remains explicit.
"""

import argparse
from pathlib import Path

import basf2 as b2
import modularAnalysis as ma
import stdPi0s
import stdV0s
from variables import variables as variable_manager


CONTINUUM_FEATURES = [
    "R2",
    "thrustBm",
    "thrustOm",
    "cosTBTO",
    "cosTBz",
    "KSFWVariables(pt_sum)",
    "KSFWVariables(mm2)",
    "KSFWVariables(hso00)",
    "KSFWVariables(hso01)",
    "KSFWVariables(hso02)",
    "KSFWVariables(hso03)",
    "KSFWVariables(hso04)",
    "KSFWVariables(hso10)",
    "KSFWVariables(hso12)",
    "KSFWVariables(hso14)",
    "KSFWVariables(hso20)",
    "KSFWVariables(hso22)",
    "KSFWVariables(hso24)",
    "KSFWVariables(hoo0)",
    "KSFWVariables(hoo1)",
    "KSFWVariables(hoo2)",
    "KSFWVariables(hoo3)",
    "KSFWVariables(hoo4)",
    *[f"CleoConeCS({index})" for index in range(1, 10)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input Belle II MDST ROOT file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Private analysis-ntuple ROOT output path",
    )
    parser.add_argument(
        "--label",
        required=True,
        type=int,
        choices=(0, 1),
        help="Binary class label: 1 for signal, 0 for continuum background",
    )
    parser.add_argument(
        "--max-events",
        default=0,
        type=int,
        help="Maximum input events to process; 0 processes all events",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input MDST does not exist: {args.input}")
    if args.max_events < 0:
        raise ValueError("--max-events must be zero or a positive integer")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    b2.set_random_seed("IDC409-continuum-preparation")
    analysis_path = b2.create_path()

    # Keep each source in its own job; the CLI label records class provenance.
    ma.inputMdst(filename=str(args.input.resolve()), path=analysis_path)

    # Match the official B2A711 construction associated with these MDST files.
    stdV0s.stdKshorts(path=analysis_path)
    stdPi0s.stdPi0s(listtype="eff40_May2020", path=analysis_path)
    ma.reconstructDecay(
        decayString="B0 -> K_S0:merged pi0:eff40_May2020",
        cut="5.2 < Mbc < 5.3 and -0.3 < deltaE < 0.2",
        path=analysis_path,
    )
    ma.matchMCTruth(list_name="B0", path=analysis_path)
    ma.buildRestOfEvent(target_list_name="B0", path=analysis_path)

    clean_mask = (
        "cleanMask",
        "nCDCHits > 0 and useCMSFrame(p) <= 3.2",
        "p >= 0.05 and useCMSFrame(p) <= 3.2",
    )
    ma.appendROEMasks(
        list_name="B0",
        mask_tuples=[clean_mask],
        path=analysis_path,
    )
    ma.buildContinuumSuppression(
        list_name="B0",
        roe_mask="cleanMask",
        path=analysis_path,
    )
    # MC truth keeps genuine signal decays but all continuum candidates.
    ma.applyCuts(list_name="B0", cut="isSignal or isContinuumEvent", path=analysis_path)

    variable_manager.addAlias("target", f"formula({args.label})")
    ma.variablesToNtuple(
        decayString="B0",
        variables=CONTINUUM_FEATURES
        + ["Mbc", "deltaE", "target"],
        treename="events",
        filename=str(args.output.resolve()),
        path=analysis_path,
    )
    b2.process(
        analysis_path,
        max_event=args.max_events,
        calculateStatistics=True,
    )
    print(b2.statistics)


if __name__ == "__main__":
    main()
