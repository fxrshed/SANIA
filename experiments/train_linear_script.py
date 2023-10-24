from train_linear import main
from utils import restricted_float
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from datasets directory.")
    parser.add_argument("--percentage", type=restricted_float, default=1.0, help="What percentage of data to use. Range from (0.0, 1.0].")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq", "mse"])
    parser.add_argument("--optimizer", type=str, choices=["psps2", "sp2plus", "adam", "adagrad", "adadelta", "sgd"])
    parser.add_argument("--hutch_init_iters", type=int, default=1000)
    args = parser.parse_args()

    if args.optimizer == "sp2plus":
        # scales = [0, 2, 4, 6, 8]
        scales = [1]
        seeds = [0, 1, 2, 3, 4]

        i = 0
        total = len(seeds) * len(scales)
        for seed in seeds:
            for scale in scales:
                print(f"{i}/{total}")
                i += 1

                print(args)
                print(seed, scale)
                main(optimizer_name=args.optimizer, loss_function_name=args.loss, dataset_name=args.dataset, 
                    batch_size=args.batch_size, percentage=args.percentage, scale=scale, 
                    epochs=args.epochs, seed=seed, save=True, lr=1.0, 
                    precond_method="none", pcg_method="none", hutch_init_iters=args.hutch_init_iters)
                print("Done!")


    elif args.optimizer == "psps2":
        # scales = [0, 2, 4, 6, 8]
        scales = [1]
        seeds = [0, 1, 2, 3, 4]
        # precond_methods = ["none", "hutch", "adam", "adam_m", "adagrad", "adagrad_m", "pcg", "scaling_vec"] 
        # pcg_methods = ["none", "hutch", "adam", "adam_m", "adagrad", "adagrad_m"]
        
        precond_methods = ["none", "hutch", "adam", "adam_m", "adagrad", "adagrad_m", "scaling_vec"]
        pcg_methods = ["none"]

        # precond_methods = ["pcg"] 
        # pcg_methods = ["none", "hutch", "adam", "adam_m", "adagrad", "adagrad_m"]
        
        i = 0
        total = len(seeds) * len(scales) * len(precond_methods) * len(pcg_methods)
        for seed in seeds:
            for scale in scales:
                for precond in precond_methods:
                    for pcg_method in pcg_methods:
                        print(f"{i}/{total}")
                        i += 1

                        print(args)
                        print(seed, scale)
                        main(optimizer_name=args.optimizer, loss_function_name=args.loss, dataset_name=args.dataset, 
                            batch_size=args.batch_size, percentage=args.percentage, scale=scale, 
                            epochs=args.epochs, seed=seed, save=True, lr=1.0, 
                            precond_method=precond, pcg_method=pcg_method, hutch_init_iters=args.hutch_init_iters)
                        print("Done!")
    else:
        lrs = [2**x for x in range(-20, 4, 2)] + [2**x for x in range(3, 6)]
        # lrs = [2**x for x in range(3, 6)]
        # scales = [0, 2, 4, 6, 8]
        scales = [1]
        seeds = [0, 1, 2, 3, 4]

        i = 0
        total = len(lrs) * len(scales) * len(seeds)
        for seed in seeds:
            for lr in lrs:
                for scale in scales:
                    print(f"{i}/{total}")
                    i += 1

                    print(args)
                    print(seed, lr, scale)
                    main(optimizer_name=args.optimizer, loss_function_name=args.loss, dataset_name=args.dataset, 
                        batch_size=args.batch_size, percentage=args.percentage, scale=scale, 
                        epochs=args.epochs, seed=seed, save=True, lr=lr, 
                        precond_method="none", pcg_method="none", hutch_init_iters=1000)
                    print("Done!")