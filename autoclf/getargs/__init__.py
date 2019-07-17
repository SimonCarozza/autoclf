import argparse

# class FooAction(argparse.Action):
#     def __init__(self, option_strings, dest, nargs=None, **kwargs):
#         if nargs is not None:
#             raise ValueError("nargs not allowed")
#         super(FooAction, self).__init__(option_strings, dest, **kwargs)
#     def __call__(self, parser, namespace, values, option_string=None):
#         print('%r %r %r' % (namespace, values, option_string))
#         setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser(prog="Autoclf",
    description="Get experiment's arguments.")
parser.add_argument("-n", "--name", nargs=1, type=str,
    help="Dataset's name as name of experiment's models and results")
parser.add_argument("--hurry", nargs=1, type=str,
    choices=["YES", "yes", "Y", "y", "NO", "no", "N", "n"],
    help="Tell %(prog)s whether you want quick evaluation")
parser.add_argument("-e", "--encode", nargs=1, type=str,
    choices=['le', 'ohe'],
    help="Encoder: [1] Label-Encoding, [2] One-Hot-Encoding")
parser.add_argument("-sc", "--scaler", nargs=1, type=int,
    choices=[1, 2, 3, 4, 5, 6],
    help="Scaler: [1] standard, [2] robust, [3] minmaxscaler,\n"
         "[4] normal scaler, [5] uniform quantile transf.,\n"
         "[6] gaussian quantile tr.\n")
parser.add_argument("-ns", "--nsplit", nargs=1, type=int,
    help="Number of folds you want to split train data into")
parser.add_argument("--npset", nargs=1, type=int,
    help="Number of number of parameter settings sampled by RSCV")
parser.add_argument("--nepoch", nargs="+", type=int,
    help="Number of epochs to train Keras NNs")
args = parser.parse_args([])


def get_args():
    return args

def get_name():
    try:
        ds_name = get_args().name[0]
    except TypeError as te:
        ds_name = None
    except Exception as e:
        raise e
    else:
        if len(ds_name) > 5:
            ds_name = ds_name[:5]
        print("Current dataset's name:", ds_name)
        print()
    return ds_name


def get_mood():
    try:
        mood = get_args().hurry[0]
    except TypeError as te:
        mood = None
    except Exception as e:
        raise e
    return mood


def get_encoder_name():
    try:
        enc = get_args().encode[0]
    except TypeError as te:
        enc = None
    except Exception as e:
        raise e
    else:
        print("Current encoder:", enc)
        print()
    return enc


def get_scaler_name():
    try:
        sc = get_args().scaler[0]
    except TypeError as te:
        sc = None
    except Exception as e:
        raise e
    else:
        print("Current scaler:", sc)
        print()
    return sc


def get_learning_mode():
    try:
        lrn_mode = get_args().mode[0]
    except TypeError as te:
        lrn_mode = None
    except Exception as e:
        raise e
    else:
        raise ValueError("'%s' is not a valid mode value. "
                         "Valid options are ['quick', "
                         "'standard', 'hard']")
    return lrn_mode


def get_n_split():
    try:
        n_split = get_args().nsplit[0]
    except TypeError as te:
        n_split = None
    except Exception as e:
        raise e
    else:
        if n_split > 20:
            n_split = 20
            print("Evaluation is gonna take forever or folds "
                  "may not have enough classes.")
            print("We set n_splits = 20 for you.")
        print()
    return n_split


def get_n_param_setting():
    try:
        n_pset = get_args().npset[0]
    except TypeError as te:
        n_pset = None
    except Exception as e:
        raise e
    else:
        if n_pset > 50:
            n_pset = 50
            print("Evaluation is gonna take forever or folds "
                  "may not have enough classes.")
            print("We set n_iter = 50 for you.")
        print()
    return n_pset


def get_n_epoch():
    n_epoch_list = get_args().nepoch

    c = 0
    # method will return a list of one or two int values <= 1000
    if n_epoch_list is not None:
        n_epoch = []
        for ep in n_epoch_list:
            print("nb_epoch:", ep)
            c += 1
            if c == 3:
                print("You don't need more than two epoch values "
                      "in this program.")
                print()
                break
            if ep > 1000:
                ep = 1000
                print("Iteration of Keras models is gonna take forever.")
                print("We set n_epoch = 1000 for you.")
                print()
            n_epoch.append(ep)
            
        n_epoch_list = n_epoch

    return n_epoch_list