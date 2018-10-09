import common
import datasets


def main():
    dataset = datasets.WikiText2
    common.process(dataset=dataset)


if __name__ == '__main__':
    main()