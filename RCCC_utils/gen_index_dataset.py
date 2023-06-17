from torch.utils.data import Dataset


class gen_index_dataset(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_image, each_label, each_true_label, index
