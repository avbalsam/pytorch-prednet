import torch
from torch.autograd import Variable


def get_half(input_tensor, half, dim=2):
    """
    Returns the input tensor with half of one dimension blacked out.

    :param input_tensor: A tensor representing an image or framesequence
    :param half: The half of the image to remove. Choose a value from ["top", "bottom"]
    :param dim: The dimension at which to split the tensor.
    :return: A tensor with the same dimensionality as the input tensor, with half of the values in one dimension set to zero.
    """
    assert half in ["top", "bottom"], "Invalid 'half' value. Choose from [\"top\", \"bottom\"]"
    dim_size = input_tensor.shape[dim]
    input_tensor_top = input_tensor[:, :, :int(dim_size/2), :]
    input_tensor_bottom = torch.zeros(10, 3, int(dim_size/2), 256)
    return torch.cat((input_tensor_top, input_tensor_bottom), dim=dim)


def get_accuracy(val_loader, model, timestep=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    correct_guesses = 0
    total_guesses = 0
    accuracy = 0
    wrong_guesses = list()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = Variable(inputs.to(device))

        if timestep is None:
            classification = model(inputs)
        else:
            classification = model(inputs, timestep)

        for j in range(len(classification)):
            c = classification[j].tolist()
            _class = c.index(max(c))
            # print(
            #    f"Image {i + j}; Correct label: {labels[j]}; Predicted: {_class}; Correct guesses: {correct_guesses}; "
            #    f"Accuracy: {round(accuracy * 100, 3)}%")
            total_guesses += 1
            if labels[j] == _class:
                correct_guesses += 1
                accuracy = correct_guesses / total_guesses
            else:
                wrong_guesses.append((labels[j], _class))

        if i % 10 == 0:
            print("Timestep: {}, Step: {}, Accuracy: {}".format("avg" if timestep is None else timestep, i, accuracy))
    return accuracy