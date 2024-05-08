import finetune
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq"])
    args = parser.parse_args()
    loss = args.loss

    for opt in ["adam", "adagrad", "adadelta"]:
        for dataset_name, batch_size in zip(["synthetic-classification-1000x1000", "colon-cancer", "duke", "leu"], [200, 16, 16, 16]):
            for scale in [0, 6]:
                print(loss, opt, dataset_name, scale)
                finetune.main(
                    dataset_name=dataset_name,
                    scale=scale,
                    batch_size=batch_size,
                    epochs=500,
                    loss=loss, 
                    opt=opt,
                    n_trials=50
                    )
                print("Done!")