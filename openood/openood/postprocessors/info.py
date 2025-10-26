num_classes_dict = {
    'cifar10': 10,
    'cifar100': 100,
    'cifar20': 20,
    'imagenet200': 200,
    'imagenet': 1000,
    'tinyimagenet':200
}
def get_num_classes(dataset):
    if "coarse" in dataset and "cifar100" in dataset:
        return 20
    return num_classes_dict[dataset.split("_")[0]]