from os import path, makedirs
from datetime import datetime

def create_train_directory(args, config_in_foldername = False):
    problem = "_".join(args.samples.name.split("/")[-1].split("_")[-2:])

    dirname = problem
    if config_in_foldername:
        dirname += f"_{args.activation}_{args.output_layer}_" + \
            f"hid{args.hidden_layers}_w{args.weight_decay}_d{args.dropout_rate}"

    dirname = args.output_folder/f"{dirname}_{datetime.now().isoformat().replace('-', '.').replace(':', '.')}"

    if path.exists(dirname):
        raise RuntimeError(f"Directory {dirname} already exists")
    makedirs(dirname)
    makedirs(dirname/"models")

    return dirname
