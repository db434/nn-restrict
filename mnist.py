import common
import datasets


def main():
    dataset = datasets.MNIST
    common.process(dataset=dataset)


if __name__ == '__main__':
    main()
