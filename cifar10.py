import common
import datasets


def main():
    dataset = datasets.Cifar10
    common.process(dataset=dataset)


if __name__ == '__main__':
    main()
