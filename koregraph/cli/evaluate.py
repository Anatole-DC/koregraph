from argparse import ArgumentParser


from koregraph.api.evaluating.loss import plot_loss


parser = ArgumentParser(
    "Koregraph evaluation",
    description="Use this to get the plot of the loss function",
)

parser.add_argument(
    "-m",
    "--model",
    dest="model_name",
    required=True,
    help="Model name",
    default="model",
)


def main():
    arguments = parser.parse_args()
    model_name = str(arguments.model_name)
    plot_loss(model_name)


if __name__ == "__main__":
    main()
