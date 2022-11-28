import torch
from torch.autograd import Variable


def get_accuracy(val_loader, model, timestep=None):
    device = model.get_device()

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