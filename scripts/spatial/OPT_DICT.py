#! /usr/bin/env python3

def options_dict(options):
    opt_dict = {"isovalue": 1.0, "outer": 0.2, "inner": 0.02, "verbose": False}

    for i, op in enumerate(options[0::2]):
        if op == "--help":
            print(
                "To run NCICLUSTER do: ./ncicluster.py input_names [OPTIONS]",
                "Options:",
                "  --isovalue i       set the isovalue to i",
                "  --outer out        set the outer limit of integration range, default 0.2",
                "  --inner in         set the inner limit of integration range, default 0.02",
                "  --mol1 m1          input molecular geometry, molecule1",
                "  --mol2 m2          input molecular geometry, molecule2",
                "  -v V               choose verbose mode, default is False",
                "  --help             display this help and exit",
                sep="\n",
            )
            exit()
        else:
            if op == "--isovalue":
                opt_dict["isovalue"] = float(options[2 * i + 1])
            elif op == "--outer":
                opt_dict["outer"] = float(options[2 * i + 1])
            elif op == "--inner":
                opt_dict["inner"] = float(options[2 * i + 1])
            elif op == "-v":
                if options[2 * i + 1] == "True":
                    opt_dict["verbose"] = True
                elif options[2 * i + 1] == "False":
                    opt_dict["verbose"] = False
                else:
                    raise ValueError(
                        "{} is not a valid option for -v. Try True or False,".format(options[2*i+1]))
            else:
                raise ValueError("{} is not a valid option".format(op))

    return opt_dict

def options_energy_calc(options):
    opt_dict = {"isovalue": 1.0, "outer": 0.2, "inner": 0.02, "gamma": 0.85, "intermol": True, "ispromol": True, "cluster": False}
    for i, op in enumerate(options[0::2]):
        if op == "--help":
            print(
                "To run NCIENERGY do: ./ncienergy.py input_names [OPTIONS]",
                "Options:",
                "  --isovalue i       set the RDG isovalue, default 1.0",
                "  --outer out        set the outer limit of integration range, default 0.20",
                "  --inner in         set the inner limit of integration range, default 0.02",
                "  --gamma g          set intermolecularity gamma value, default is 0.85",
                "  --intermol im      determine if intermolecular mode is on, default True",
                "  --ispromol p       determine if promolecular mode is on, default True",
                "  --clustering c     determine if clustering mode in on, default False",
                "  --help             display this help and exit",
                sep="\n",
            )
            exit()
        else:
            if op == "--isovalue":
                opt_dict["isovalue"] = float(options[2 * i + 1])
            elif op == "--outer":
                opt_dict["outer"] = float(options[2 * i + 1])
            elif op == "--inner":
                opt_dict["inner"] = float(options[2 * i + 1])
            elif op == "--gamma":
                opt_dict["gamma"] = float(options[2 * i + 1])
            elif op == "--intermol":
                if options[2 * i + 1] == "F":
                    opt_dict["intermol"] = False
            elif op == "--ispromol":
                if options[2 * i + 1] == "F":
                    opt_dict["ispromol"] = False
            elif op == "--clustering":
                if options[2 * i + 1] == "T":
                    opt_dict["cluster"] = True
            elif op == "--mol1":
                opt_dict["mol1"] = options[2 * i + 1]
            elif op == "--mol2":
                opt_dict["mol2"] = options[2 * i + 1]
            else:
                raise ValueError("{} is not a valid option".format(op))
    return opt_dict