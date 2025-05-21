#! /usr/bin/env python3

def options_dict(options):
    opt_dict = {"isovalue": 1.0, "outer": 0.2, "inner": 0.02, "verbose": False}

    for i, op in enumerate(options[0::2]):
        if op == "--help":
            print(
                "To run NCICLUSTER do: ./ncicluster.py input_names [OPTIONS]",
                "Options:",
                "  --isovalue i       set the isovalue to i",
                "  --outer out        set the outer limit of integration range, default 0.07",
                "  --inner in         set the inner limit of integration range, default 0.01",
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
