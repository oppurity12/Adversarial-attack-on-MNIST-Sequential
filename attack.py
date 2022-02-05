import torch
import torch.nn as nn

from tqdm import tqdm


def fgsm(data, epsilon):
    perturbation = data.grad.data.sign()
    perturbed_data = data + epsilon * perturbation
    return perturbed_data


def fg_main(model, data_loader, epsilon, gpus):
    total_batch = len(data_loader)
    device = 'cuda' if gpus > 0 else 'cpu'
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_accuracy = 0
    adv_accuracy = 0
    for x, y in tqdm(data_loader, desc="attack model"):
        x = x.to(device)
        y = y.to(device)
        x = x.permute(1, 0, 2).contiguous()
        x.requires_grad = True
        output = model(x)

        prediction = (torch.argmax(output, dim=1) == y).float().mean().item()
        test_accuracy += prediction

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()

        perturbed_data = fgsm(x, epsilon)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        perturbed_output = model(perturbed_data)

        adv_prediction = (torch.argmax(perturbed_output, dim=1) == y).float().mean().item()

        adv_accuracy += adv_prediction

    adv_accuracy = (adv_accuracy / total_batch) * 100

    test_accuracy = (test_accuracy / total_batch) * 100
    print(f"test_accuracy: {test_accuracy:.3f}%")
    print(f"fgsm_accuracy: {adv_accuracy:.3f}%")

    return adv_accuracy, test_accuracy


def pgd_main(model, data_loader, epsilon, gpus, attack_steps):
    adv_accuracy = 0
    total_batch = len(data_loader)
    device = 'cuda' if gpus > 0 else 'cpu'
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    for x, y in tqdm(data_loader, desc="attack model"):
        test_accuracy = 0
        for step in range(attack_steps):
            x = x.detach().clone()
            y = y.detach().clone()

            x = x.to(device)
            y = y.to(device)
            if step == 0:
                x = x.permute(1, 0, 2).contiguous()
            x.requires_grad = True
            output = model(x)

            if step == 0:
                prediction = (torch.argmax(output, dim=1) == y).float().mean().item()
                test_accuracy += prediction

            loss = criterion(output, y)
            model.zero_grad()
            loss.backward()

            perturbed_data = fgsm(x, epsilon)
            x = torch.clamp(perturbed_data, 0, 1)

        perturbed_output = model(x)
        adv_prediction = (torch.argmax(perturbed_output, dim=1) == y).float().mean().item()

        adv_accuracy += adv_prediction

    adv_accuracy = (adv_accuracy / total_batch) * 100

    test_accuracy = (test_accuracy / total_batch) * 100
    print(f"test_accuracy: {test_accuracy:.3f}%")

    print(f"pgd_accuracy: {adv_accuracy:.3f}%")

    return adv_accuracy, test_accuracy