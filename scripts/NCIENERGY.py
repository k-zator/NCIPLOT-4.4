#! /usr/bin/env python3

import sys
from spatial.OPT_DICT import options_energy_calc
from spatial.calculate_NCI_energy import (
    calculate_energy_cluster,
    calculate_energy_single,
    calculate_charge_correction,
    calculate_charge_correction_cluster,
)

"""
Calculate binding energy from the NCIPLOT analysis. Crucially, only calculate it when the correct parameters are set.
Otherwise, print a message saying it only works for the correct set.
Implemented for the NCICLUSTER results so that a single function could be used for any complex, 
including ones where sigma hole interactions are present. 
Parameters: path to nci_output_file. (afaik, it produces the values in the file in previous step so this is sort of recursive)
"""

def _print_default_parameter_message(include_range_n=False):
    print(" NCIENERGY mode runs only with the default parameters:")
    print(" RDG_CUTOFF 1.00 0.30")
    print(" INTEGRATE")
    print(" INTERMOL_CUTOFF 0.85 (0.75 for DFT)")
    if include_range_n:
        print(" RANGE 3")
    else:
        print(" RANGE")
    print(" -0.20 -0.02")
    print(" -0.02  0.02")
    print("  0.02  0.20")


def _validate_default_parameters(opt_dict):
    is_default_core = (
        opt_dict["isovalue"] == 1.0
        and opt_dict["outer"] == 0.2
        and opt_dict["inner"] == 0.02
        and opt_dict["intermol"] is True
    )
    if not is_default_core:
        _print_default_parameter_message(include_range_n=False)
        return False

    gamma = opt_dict["gamma"]
    ispromol = opt_dict["ispromol"]
    if (ispromol and gamma == 0.85) or ((not ispromol) and gamma == 0.75):
        return True

    _print_default_parameter_message(include_range_n=True)
    return False


def _parse_cli(argv):
    if not argv:
        print("Usage: NCIENERGY.py input_names [OPTIONS]")
        return None, ["--help"]

    options = []
    if argv[0] != "--help":
        input_name = argv[0]
    else:
        input_name = None
        options.append("--help")

    if len(argv) > 1:
        options += argv[1:]

    return input_name, options


def main(argv=None):
    input_name, options = _parse_cli(sys.argv[1:] if argv is None else argv)
    opt_dict = options_energy_calc(options)

    if input_name is None:
        return 0

    if not _validate_default_parameters(opt_dict):
        return 1

    oname = opt_dict["oname"]
    ispromol = opt_dict["ispromol"]
    cluster = opt_dict["cluster"]
    supra = opt_dict["supra"]
    mol1 = opt_dict["mol1"]
    mol2 = opt_dict["mol2"]
    total_charges = opt_dict["total_charges"]
    use_charges = opt_dict["use_charges"]

    files = []
    with open(input_name, "r") as f:
        for line in f:
            files.append(line[:-1])
    filename = files[0]

    try:
        with open(f"nci_{oname}.out") as f:
            contents = f.readlines()
    except FileNotFoundError:
        print(f"Error: nci_{oname}.out not found. Please ensure that your NCIPLOT output file is named 'nci_{oname}.out'.")
        return 1

    print("----------------------------------------------------------------------")
    print("                             NCIENERGY                                ")
    print("----------------------------------------------------------------------")
    if ispromol:
        print(" Calculating energy using the promolecular equations")
        use_charge_model = use_charges
        if cluster:
            if use_charge_model:
                E_charge_correction, E_charge_clusters = calculate_charge_correction_cluster(
                    mol1,
                    mol2,
                    ispromol,
                    total_charges,
                    filename + "_CPs.xyz",
                    cutoff=7.0,
                    supra=supra,
                    return_shapley=True,
                )
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol, supra, mol1, mol2, filename)
            for cluster_id, (e_sum, e_polar, e_vdw) in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f" Cluster {cluster_id} energies / kJ/mol")
                if use_charge_model and cluster_id < len(E_charge_clusters):
                    print(" E_sum    :        {:.2f}".format(e_sum + E_charge_clusters[cluster_id]))
                else:
                    print(" E_sum    :        {:.2f}".format(e_sum))
                print(" E_polar  :        {:.2f}".format(e_polar))
                print(" E_vdw    :        {:.2f}".format(e_vdw))
                if use_charge_model and cluster_id < len(E_charge_clusters):
                    print(" E_charge :        {:.2f}".format(E_charge_clusters[cluster_id]))
                elif use_charge_model:
                    print(" E_charge :        n/a")
                print("----------------------------------------------------------------------")
            if use_charge_model and len(E_charge_clusters) != len(E_sum):
                print(" Warning: cluster count mismatch between NCI integration and charge attribution vectors.")
            print(" Summed-across-clusters integrals / kJ/mol")
            if use_charge_model:
                print(" E_sum    :        {:.2f}".format(E_sum.sum() + E_charge_correction))
            else:
                print(" E_sum    :        {:.2f}".format(E_sum.sum()))
            print(" E_polar  :        {:.2f}".format(E_polar.sum()))
            print(" E_vdw    :        {:.2f}".format(E_vdw.sum()))
            if use_charge_model:
                print(" E_charge :        {:.2f}".format(E_charge_correction))
            print("----------------------------------------------------------------------")
        else:
            if use_charge_model:
                E_charge_correction = calculate_charge_correction(mol1, mol2, ispromol, total_charges, supra=supra)
            print(" If your system contains sigma hole interactions, "
                  "consider using the clustering mode for better energy accuracy")
            print("----------------------------------------------------------------------")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol, supra)
            print(" NCI energies / kJ/mol")
            if use_charge_model:
                print(" E_sum    :        {:.2f}".format(E_sum + E_charge_correction))
            else:
                print(" E_sum    :        {:.2f}".format(E_sum))
            print(" E_polar  :        {:.2f}".format(E_polar))
            print(" E_vdw    :        {:.2f}".format(E_vdw))
            if use_charge_model:
                print(" E_charge :        {:.2f}".format(E_charge_correction))
            print("----------------------------------------------------------------------")

    else:
        print(" Calculating energy using the DFT equations")
        if cluster:
            E_sum, E_polar, E_vdw = calculate_energy_cluster(contents, ispromol, supra, mol1, mol2, filename)
            for cluster_id, (e_sum, e_polar, e_vdw) in enumerate(zip(E_sum, E_polar, E_vdw)):
                print(f" Cluster {cluster_id} energies / kJ/mol")
                print(" E_sum   :        {:.2f}".format(e_sum))
                print(" E_polar :        {:.2f}".format(e_polar))
                print(" E_vdw   :        {:.2f}".format(e_vdw))
                print("----------------------------------------------------------------------")
            print(" Summed-across-clusters integrals / kJ/mol")
            print(" E_sum   :        {:.2f}".format(E_sum.sum()))
            print(" E_polar :        {:.2f}".format(E_polar.sum()))
            print(" E_vdw   :        {:.2f}".format(E_vdw.sum()))
            print("----------------------------------------------------------------------")
        else:
            print(" If your system contains sigma hole interactions, "
                  "consider using the clustering mode for better energy accuracy")
            print("----------------------------------------------------------------------")
            E_sum, E_polar, E_vdw = calculate_energy_single(contents, ispromol, supra)
            print(" NCI energies / kJ/mol")
            print(" E_sum   :        {:.2f}".format(E_sum))
            print(" E_polar :        {:.2f}".format(E_polar))
            print(" E_vdw   :        {:.2f}".format(E_vdw))
            print("----------------------------------------------------------------------")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
