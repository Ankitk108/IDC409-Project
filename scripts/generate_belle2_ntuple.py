#!/usr/bin/env python3
"""Generate an unsplit continuum-suppression ntuple with official Belle II MC.

Run this script only inside a configured Belle II basf2 environment. The generated
ROOT file is private input data and must be kept outside Git.
"""

import argparse

import basf2 as b2
import modularAnalysis as ma
import stdPi0s


CONTINUUM_FEATURES = [
    "R2",
    "thrustBm",
    "thrustOm",
    "cosTBTO",
    "cosTBz",
    "KSFWVariables(et)",
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
        "--output",
        default="belle2_continuum.root",
        help="Private ROOT ntuple output path (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_path = b2.create_path()

    # Official starterkit samples: one signal decay and two continuum flavors.
    sample_paths = [
        b2.find_file("starterkit/2021/B02ks0pi0_sample.root", data_type="examples"),
        b2.find_file("starterkit/2021/uubar_sample.root", data_type="examples"),
        b2.find_file("starterkit/2021/ccbar_sample.root", data_type="examples"),
    ]
    ma.inputMdstList(filelist=sample_paths, path=analysis_path)

    stdPi0s.stdPi0s(listtype="eff60_May2020", path=analysis_path)
    ma.fillParticleList(
        decayString="pi+:good",
        cut="chiProb > 0.001 and pionID > 0.5",
        path=analysis_path,
    )
    ma.reconstructDecay(
        decayString="K_S0 -> pi+:good pi-:good",
        cut="0.480 <= M <= 0.516",
        path=analysis_path,
    )
    ma.reconstructDecay(
        decayString="B0 -> K_S0 pi0:eff60_May2020",
        cut="5.1 < Mbc < 5.3 and abs(deltaE) < 2",
        path=analysis_path,
    )
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

    # isNotContinuumEvent is 1 for signal and 0 for continuum background.
    ma.variablesToNtuple(
        decayString="B0",
        variables=CONTINUUM_FEATURES + ["Mbc", "deltaE", "isNotContinuumEvent"],
        treename="events",
        filename=args.output,
        path=analysis_path,
    )
    b2.process(analysis_path)
    print(b2.statistics)


if __name__ == "__main__":
    main()

