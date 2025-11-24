from fire import Fire

from u2.diagnostic import sanity_check
from u2.inference import predict


def main():

    Fire({
        "sanity_check": sanity_check,
        "predict": predict
    })


if __name__ == "__main__":
    main()
