# This is a process to find sensitive filters corresponding classes
# !/usr/bin/env python3
import argparse
import copy
import torch
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader_all, UniformNormLossFunc
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.inf)

@torch.no_grad()
def test(model, test_loader):
    model.eval()
    train_correct_1 = 0.0
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            train_correct_1 += correct[:, :1].sum()
    return float(train_correct_1)/len(test_loader.dataset)


@torch.no_grad()
def test_class(model_args, model, test_loader):
    '''

    :param model_args: model arguments
    :param model:
    :param test_loader:
    :return:
    '''
    model.eval()
    train_correct_1 = 0.0
    ground_truth_all = None
    inference_result_all = None
    test_acc_by_class = []

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            if model_args.gpu:
                image = image.to(settings.CUDA)
                label = label.to(settings.CUDA)

            output = model(image)
            _, pred = output.topk(1, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            train_correct_1 += correct[:, :1].sum()

            if ground_truth_all is None:
                ground_truth_all = label.to('cpu')
            else:
                ground_truth_all = torch.cat((ground_truth_all, label.to('cpu')))

            if inference_result_all is None:
                inference_result_all = pred.to('cpu')
            else:
                inference_result_all = torch.cat((inference_result_all, pred.to('cpu')))

    for c in range(n_classes):
        correct_number = 0
        ground_truth_index = torch.where(ground_truth_all == int(c))[0]
        inference_index = torch.where(inference_result_all == int(c))[0]

        for i in inference_index:
            if i in ground_truth_index:
                correct_number += 1

        test_acc_by_class.append(correct_number/len(ground_truth_index))

    return float(train_correct_1) / len(test_loader.dataset), test_acc_by_class


def get_ground_truth_prediction(model_args, model, data_loader):
    ground_truth = []
    prediction = []

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            if model_args.gpu:
                images = images.to(settings.CUDA)
                labels = labels.to(settings.CUDA)

            outputs = model(images)
            ground_truth.extend(labels.tolist())

            _, predicts = outputs.max(1)
            prediction.extend(predicts.tolist())

            del images, labels
            if model_args.gpu:
                with torch.cuda.device(settings.CUDA):
                    torch.cuda.empty_cache()

    return ground_truth, prediction


def sensitive_filters(model_args, model, test_loader, mutual_classes, noise_var=0.5):
    '''
    So this is where a key distinction between terms comes in handy:
    whereas in the 1 channel case, where the term filter and kernel are interchangeable. In the general case,
    they’re actually pretty different. Each filter actually happens to be a collection of kernels, with there
    being one kernel for every single input channel to the layer, and each kernel being unique.

    :param model_args: model arguments
    :param model:
    :param test_loader:
    :param mutual_classes:
    :param noise_var:
    :return:
    '''

    sensitive_filters_all_file = 'results_mobilenet_cifar100/sensitive_filters_all_' + str(noise_var) + '.txt'
    sensitive_filters_each_class_file = 'results_mobilenet_cifar100/sensitive_filters_each_class_' + str(noise_var) + '.txt'
    # part 1: find the sensitive filters for all the classes
    sensitive_filters_all = []
    if os.path.exists(sensitive_filters_all_file):
        file = open(sensitive_filters_all_file, 'r')
        sensitive_filters_all = eval(file.read())
    else:
        original_avg_test_acc, original_test_acc_class = test_class(model_args=model_args, model=model, test_loader=test_loader)
        print("Average test accuracy is: ", original_avg_test_acc)
        print("Test accuracy for each class: ", original_test_acc_class)

        filter_number_all = 0
        filter_number_affected = 0
        # torch.randn produces a tensor with elements drawn from a Gaussian distribution of zero mean and unit variance
        # Multiply by 0.05 to have the desired variance
        state_dict_original = torch.load(model_args.weights, map_location=settings.CUDA)
        for layer in state_dict_original:
            layer_size = state_dict_original[layer].size()
            if len(layer_size) == 4:
                print(layer, '\t', state_dict_original[layer].size())
                filter_number_all += layer_size[0]
                # for filter_index in range(layer_size[0]):
                #     noise = noise_var * torch.randn(layer_size[1], layer_size[2], layer_size[3])
                #     state_dict_temp = torch.load(model_args.weights, map_location=settings.CUDA)
                #     state_dict_temp[layer][filter_index] += noise.to(settings.CUDA)
                #
                #     model.load_state_dict(state_dict_temp)
                #     avg_test_acc, test_acc_class = test_class(model_args=model_args, model=model, test_loader=test_loader)
                #
                #     test_acc_diff = original_avg_test_acc - avg_test_acc
                #     test_acc_diff_class = torch.tensor(original_test_acc_class) - torch.tensor(test_acc_class)
                #
                #     # absolute sort
                #     # sorted_values, sorted_indices = torch.sort(torch.abs(torch.tensor(test_acc_diff_class)), descending=True)
                #     sorted_values, sorted_indices = torch.sort(test_acc_diff_class, descending=True)
                #
                #     if sorted_values[0] == 0 and sorted_values[-1] == 0:
                #         break
                #
                #     filter_number_affected += 1
                #     sensitive_filters_all_temp = {"layer": layer, "filter_index": filter_index, "test_acc_diff": test_acc_diff, "test_acc_diff_class": test_acc_diff_class.tolist()}
                #     sensitive_filters_all.append(sensitive_filters_all_temp)

        print(f"filter_number_all is {filter_number_all}, filter_number_affected is {filter_number_affected}")
        file = open(sensitive_filters_all_file, "w")
        file.write(str(sensitive_filters_all))
        file.close()

    # part 2: select the sensitive filters for specific classes
    sensitive_filters_each_class = []
    if os.path.exists(sensitive_filters_each_class_file):
        file = open(sensitive_filters_each_class_file, 'r')
        sensitive_filters_each_class = eval(file.read())
    else:
        for mutual_class in mutual_classes:
            # if the changes of the two classes are 0, we should discard this filter for current level noise directly
            # if the largest change is 0, we should find the first non-zero largest value
            # if the smallest change is 0, we should find the last non-zero smallest value
            sensitive_filters_class_both = []
            sensitive_filters_class_0 = []
            sensitive_filters_class_1 = []
            for sensitive_filter in sensitive_filters_all:
                if sensitive_filter['test_acc_diff_class'][mutual_class[0]] == 0 and sensitive_filter['test_acc_diff_class'][mutual_class[1]] == 0:
                    continue

                if sensitive_filter['test_acc_diff_class'][mutual_class[0]] != 0 and sensitive_filter['test_acc_diff_class'][mutual_class[1]] != 0:
                    # and abs(sensitive_filter['test_acc_diff_class'][mutual_class[0]] - sensitive_filter['test_acc_diff_class'][mutual_class[1]]) > 0: # Need difference threshould?
                    # print(({'layer': sensitive_filter['layer'], 'filter_index': sensitive_filter['filter_index']}))
                    sensitive_filters_class_both.append({'layer': sensitive_filter['layer'], 'filter_index': sensitive_filter['filter_index']})

                elif sensitive_filter['test_acc_diff_class'][mutual_class[0]] == 0:
                    sensitive_filters_class_1.append({'layer': sensitive_filter['layer'], 'filter_index': sensitive_filter['filter_index']})

                elif sensitive_filter['test_acc_diff_class'][mutual_class[1]] == 0:
                    sensitive_filters_class_0.append({'layer': sensitive_filter['layer'], 'filter_index': sensitive_filter['filter_index']})

            sensitive_filters_each_class.append({'mutual_class': mutual_class, 'sensitive_filters_class_both': sensitive_filters_class_both,
                'sensitive_filters_class_0': sensitive_filters_class_0, 'sensitive_filters_class_1': sensitive_filters_class_1})

        file = open(sensitive_filters_each_class_file, "w")
        file.write(str(sensitive_filters_each_class))
        file.close()
    return sensitive_filters_all, sensitive_filters_each_class


def freeze_filters(model_state_dict, filters_for_aspect, reverse):
    '''
    if reverse is False, it freezes the filters in the filters_for_aspects;
    if reverse is True, it freezes the other filters except in the filters_for_aspects;
    if reverse is None, it unfreezes all the filters
    :param model_state_dict: the state dictionary of current model
    :param filters_for_aspects: a list of filters to be frozen or unfrozen
    :param reverse: freeze the filters in filters_for_aspects or the other filters not in filters_for_aspects
    :return: the freezed model state dictionary
    '''
    # for layer in model.named_parameters():
    #     layer_size = len(layer[1].shape)
    #     for filter_index, matrix in enumerate(layer[1]):
    #         if layer_size == 4:
    #             if {'layer': layer[0], 'filter_index': filter_index} in filters_for_aspect:
    #                 matrix.requires_grad = True
    #             else:
    #                 print(matrix.grad_fn)
    #                 matrix._grad_fn = None

    new_model_state_dict = copy.deepcopy(model_state_dict)
    for layer in model_state_dict:
        layer_size = model_state_dict[layer].size()
        if len(layer_size) == 4:
            for filter_index in range(layer_size[0]):
                if reverse is False:
                    if {'layer': layer, 'filter_index': filter_index} in filters_for_aspect:
                        new_model_state_dict[layer][filter_index].requires_grad = False
                    else:
                        new_model_state_dict[layer][filter_index].requires_grad = True
                elif reverse is True:
                    if {'layer': layer, 'filter_index': filter_index} in filters_for_aspect:
                        new_model_state_dict[layer][filter_index].requires_grad = True
                    else:
                        new_model_state_dict[layer][filter_index].requires_grad = False
                elif reverse is None:
                    new_model_state_dict[layer][filter_index].requires_grad = True
    return new_model_state_dict


def controlled_training(model_args, model, train_loader, test_loader, mutual_classes, filters_for_aspects, noise_var=0.05):
    '''
    1. For each training batch:
        1.1 divide the batch into two parts, data corresponding to the mutual classes, and
        1.2 Only freeze the other filters that are not in filters_for_aspects and unlearn the images corresponding to the mutual classes
        1.3 Only freeze the filters that are in filters_for_aspects and train the images corresponding to the mutual classes
    2. May need some further training or patching
    :param model_args: model arguments
    :param model:
    :param train_loader:
    :param test_loader:
    :param mutual_classes:
    :param filters_for_aspects:
    :return:
    '''
    model_state_dicts = []
    sf = torch.nn.Softmax(dim=1)
    loss_function_ce = nn.CrossEntropyLoss()
    loss_function_norm = UniformNormLossFunc()

    average_acc_all_epoch = []
    test_acc_class_diff_all_epoch = []

    mutual_mis_original = []
    mutual_mis_all_epoch = []

    train_acc = test(model=model, test_loader=train_loader)
    test_acc = test(model=model, test_loader=test_loader)
    original_avg_test_acc, original_test_acc_class = test_class(model_args=model_args, model=model, test_loader=test_loader)
    original_ground_truth_list, original_prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)
    original_conf_matrix = confusion_matrix(original_ground_truth_list, original_prediction_list)

    for mutual_class in mutual_classes:
        model.load_state_dict(torch.load(model_args.weights, map_location=settings.CUDA))
        mutual_mis_original.append({str(mutual_class[0])+'_to_'+str(mutual_class[1]): original_conf_matrix[mutual_class[0]][mutual_class[1]],
                                    str(mutual_class[1])+'_to_'+str(mutual_class[0]): original_conf_matrix[mutual_class[1]][mutual_class[0]]})
        print(f"mutual_class: {mutual_class}, original: train_acc: {train_acc}, test_acc: {test_acc} "
              f"{str(mutual_class[0])}_to_{str(mutual_class[0])}: {original_conf_matrix[mutual_class[0]][mutual_class[0]]}, "
              f"{str(mutual_class[1])}_to_{str(mutual_class[1])}: {original_conf_matrix[mutual_class[1]][mutual_class[1]]}, "
              f"{str(mutual_class[0])}_to_{str(mutual_class[1])}: {original_conf_matrix[mutual_class[0]][mutual_class[1]]}, "
              f"{str(mutual_class[1])}_to_{str(mutual_class[0])}: {original_conf_matrix[mutual_class[1]][mutual_class[0]]}")

        filters_for_aspect = None
        for current in filters_for_aspects:
            if current['mutual_class'] == mutual_class:
                filters_for_aspect = current['sensitive_filters_class_both']

        average_acc_all_epoch_temp = []
        test_acc_class_diff_all_epoch_temp = []
        mutual_mis_all_epoch_temp = []

        # 1. var:0.05 unlearn:0.001 train:0.01 is not very good - 40 74 24 9 /
        # 2. var:0.05 unlearn:0.0001 train:0.001 is
        unlearn_rate = 0.0001
        train_rate = 0.001
        unlearn_optimizer = optim.SGD(model.parameters(), lr=unlearn_rate, momentum=0.9, weight_decay=1e-4)
        train_optimizer = optim.SGD(model.parameters(), lr=train_rate, momentum=0.9, weight_decay=1e-4)
        for epoch in range(1, 26):
            for index, (images, labels) in enumerate(train_loader):
                # 1. divide the batch into two parts
                images_mutual_class = None

                for i, label in enumerate(labels):
                    if label in mutual_class:
                        if images_mutual_class is None:
                            images_mutual_class = images[i].unsqueeze(0)
                        else:
                            images_mutual_class = torch.cat((images_mutual_class, images[i].unsqueeze(0)))

                if model_args.gpu:
                    images = images.to(settings.CUDA)
                    labels = labels.to(settings.CUDA)
                    if images_mutual_class is not None:
                        images_mutual_class = images_mutual_class.to(settings.CUDA)
                        uniform_label = sf(torch.full([len(images_mutual_class), model_args.num_class], 1/model_args.num_class)).to(settings.CUDA)

                # 2. freeze all filters and unfreeze both sensitive filters, and then unlearn
                if images_mutual_class is not None:
                    original_model_state_dict = copy.deepcopy(model.state_dict())
                    unlearn_optimizer.zero_grad()
                    outputs = sf(model(images_mutual_class))
                    loss = loss_function_norm(outputs, uniform_label)
                    loss.backward()
                    unlearn_optimizer.step()

                    new_model_state_dict = copy.deepcopy(model.state_dict())
                    for layer in original_model_state_dict:
                        layer_size = original_model_state_dict[layer].size()
                        if len(layer_size) == 4:
                            for filter_index in range(layer_size[0]):
                                if {'layer': layer, 'filter_index': filter_index} not in filters_for_aspect:
                                    new_model_state_dict[layer][filter_index] = original_model_state_dict[layer][filter_index]

                    # 3. freeze all filters and unfreeze other filters, and then train
                    for sub_epoch in range(1, 11):
                        model.load_state_dict(new_model_state_dict)
                        train_optimizer.zero_grad()
                        outputs = model(images)
                        loss = loss_function_ce(outputs, labels)
                        _, predicts = outputs.max(1)
                        loss.backward()
                        train_optimizer.step()

            train_acc = test(model=model, test_loader=train_loader)
            test_acc = test(model=model, test_loader=test_loader)
            avg_test_acc, test_acc_class = test_class(model_args=model_args, model=model, test_loader=test_loader)
            average_acc_all_epoch_temp.append(original_avg_test_acc-avg_test_acc)
            test_acc_class_diff_all_epoch_temp.append((np.array(original_test_acc_class)-np.array(test_acc_class)).tolist())

            ground_truth_list, prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)
            conf_matrix = confusion_matrix(ground_truth_list, prediction_list)

            if conf_matrix[mutual_class[0]][mutual_class[0]] + conf_matrix[mutual_class[1]][mutual_class[1]] + 5 > \
                    original_conf_matrix[mutual_class[0]][mutual_class[0]] + original_conf_matrix[mutual_class[1]][mutual_class[1]]\
                    and conf_matrix[mutual_class[0]][mutual_class[1]] + conf_matrix[mutual_class[1]][mutual_class[0]] < \
                    original_conf_matrix[mutual_class[0]][mutual_class[1]] + original_conf_matrix[mutual_class[1]][mutual_class[0]]:
                mutual_mis_all_epoch_temp.append({'mutual_class': mutual_class, 'epoch': epoch,
                                                  str(mutual_class[0])+'_to_'+str(mutual_class[0]): conf_matrix[mutual_class[0]][mutual_class[0]],
                                                  str(mutual_class[1])+'_to_'+str(mutual_class[1]): conf_matrix[mutual_class[1]][mutual_class[1]],
                                                  str(mutual_class[0])+'_to_'+str(mutual_class[1]): conf_matrix[mutual_class[0]][mutual_class[1]],
                                                  str(mutual_class[1])+'_to_'+str(mutual_class[0]): conf_matrix[mutual_class[1]][mutual_class[0]]})
                torch.save(model.state_dict(), 'results_mobilenet_cifar100/mutual_class_' + str(mutual_class) + '_noise_var_' + str(noise_var) + '_epoch_' + str(epoch) + '.pth')

            print(f"mutual_class: {mutual_class}, epoch: {epoch}, train_acc: {train_acc}, test_acc: {test_acc} "
                  f"{str(mutual_class[0])}_to_{str(mutual_class[0])}: {conf_matrix[mutual_class[0]][mutual_class[0]]}, "
                  f"{str(mutual_class[1])}_to_{str(mutual_class[1])}: {conf_matrix[mutual_class[1]][mutual_class[1]]}, "
                  f"{str(mutual_class[0])}_to_{str(mutual_class[1])}: {conf_matrix[mutual_class[0]][mutual_class[1]]}, "
                  f"{str(mutual_class[1])}_to_{str(mutual_class[0])}: {conf_matrix[mutual_class[1]][mutual_class[0]]}")

        average_acc_all_epoch.append(average_acc_all_epoch_temp)
        test_acc_class_diff_all_epoch.append(test_acc_class_diff_all_epoch_temp)
        mutual_mis_all_epoch.append(mutual_mis_all_epoch_temp)
        model_state_dicts.append(model.state_dict())

        file = open('results_mobilenet_cifar100/mutual_mis_all_epoch_var_' + str(noise_var) + '_unlearn_rate_' + str(unlearn_rate) + '_train_rate_' + str(train_rate) + '.txt', "w")
        file.write(str(mutual_mis_all_epoch))
        file.close()
    return model_state_dicts


def patching_old(model_args, original_model, model_state_dict_all, mutual_classes, filters_for_aspects, train_loader, test_loader):
    train_acc = test(model=original_model, test_loader=train_loader)
    test_acc = test(model=original_model, test_loader=test_loader)
    original_avg_test_acc, original_test_acc_class = test_class(model_args=model_args, model=original_model, test_loader=test_loader)
    original_ground_truth_list, original_prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)
    original_conf_matrix = confusion_matrix(original_ground_truth_list, original_prediction_list)

    for i, state_dict_temp in enumerate(model_state_dict_all):
        for mutual_class in mutual_classes:
            print(f"Original model: mutual_class: {mutual_class}, original_avg_test_acc: {original_avg_test_acc}, "
                  f"original: train_acc: {train_acc}, test_acc: {test_acc}, "
                  f"{str(mutual_class[0])}_to_{str(mutual_class[0])}: {original_conf_matrix[mutual_class[0]][mutual_class[0]]}, "
                  f"{str(mutual_class[1])}_to_{str(mutual_class[1])}: {original_conf_matrix[mutual_class[1]][mutual_class[1]]}, "
                  f"{str(mutual_class[0])}_to_{str(mutual_class[1])}: {original_conf_matrix[mutual_class[0]][mutual_class[1]]}, "
                  f"{str(mutual_class[1])}_to_{str(mutual_class[0])}: {original_conf_matrix[mutual_class[1]][mutual_class[0]]}")

        mutual_class = mutual_classes[i]
        filters_for_aspect = None
        for current in filters_for_aspects:
            if current['mutual_class'] == mutual_class:
                filters_for_aspect = current['sensitive_filters_class_both']

        if i == 0:
            original_model.load_state_dict(state_dict_temp)
        else:
            original_model_state_dict = original_model.state_dict()
            for layer in original_model_state_dict:
                layer_size = original_model_state_dict[layer].size()
                if len(layer_size) == 4:
                    for filter_index in range(layer_size[0]):
                        if {'layer': layer, 'filter_index': filter_index} not in filters_for_aspect:
                            original_model_state_dict[layer][filter_index] = state_dict_temp[layer][filter_index]

            original_model.load_state_dict(original_model_state_dict)

        train_optimizer = optim.SGD(original_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        loss_function_ce = nn.CrossEntropyLoss()
        for epoch in range(1, 11):
            if i != 0:
                original_model_state_dict = original_model.state_dict()
                for layer in original_model_state_dict:
                    layer_size = original_model_state_dict[layer].size()
                    if len(layer_size) == 4:
                        for filter_index in range(layer_size[0]):
                            if {'layer': layer, 'filter_index': filter_index} not in filters_for_aspect:
                                original_model_state_dict[layer][filter_index] = state_dict_temp[layer][filter_index]

                for index, (images, labels) in enumerate(train_loader):
                    if model_args.gpu:
                        images = images.to(settings.CUDA)
                        labels = labels.to(settings.CUDA)

                    train_optimizer.zero_grad()
                    outputs = original_model(images)
                    loss = loss_function_ce(outputs, labels)
                    loss.backward()
                    train_optimizer.step()
                train_acc = test(model=original_model, test_loader=train_loader)
                test_acc = test(model=original_model, test_loader=test_loader)
                avg_test_acc, test_acc_class = test_class(model_args=model_args, model=original_model, test_loader=test_loader)
                ground_truth_list, prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)
                conf_matrix = confusion_matrix(ground_truth_list, prediction_list)
            else:
                train_acc = test(model=original_model, test_loader=train_loader)
                test_acc = test(model=original_model, test_loader=test_loader)
                avg_test_acc, test_acc_class = test_class(model_args=model_args, model=original_model, test_loader=test_loader)
                ground_truth_list, prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)
                conf_matrix = confusion_matrix(ground_truth_list, prediction_list)
                break

            for mutual_class in mutual_classes:
                print(f"Patched model: {i}, epoch: {epoch}, mutual_class: {mutual_class}, epoch: {epoch}, avg_test_acc: {avg_test_acc}, "
                      f"train_acc: {train_acc}, test_acc: {test_acc}, "
                      f"{str(mutual_class[0])}_to_{str(mutual_class[0])}: {conf_matrix[mutual_class[0]][mutual_class[0]]}, "
                      f"{str(mutual_class[1])}_to_{str(mutual_class[1])}: {conf_matrix[mutual_class[1]][mutual_class[1]]}, "
                      f"{str(mutual_class[0])}_to_{str(mutual_class[1])}: {conf_matrix[mutual_class[0]][mutual_class[1]]}, "
                      f"{str(mutual_class[1])}_to_{str(mutual_class[0])}: {conf_matrix[mutual_class[1]][mutual_class[0]]}")


def patching(model_args, original_model, model_state_dict_all, mutual_classes, filters_for_aspects, train_loader, test_loader):
    train_acc = test(model=original_model, test_loader=train_loader)
    test_acc = test(model=original_model, test_loader=test_loader)
    original_avg_test_acc, original_test_acc_class = test_class(model_args=model_args, model=original_model, test_loader=test_loader)
    original_ground_truth_list, original_prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)

    original_conf_matrix = confusion_matrix(original_ground_truth_list, original_prediction_list)

    for mutual_class in mutual_classes:
        print(f"Original model: mutual_class: {mutual_class}, original_avg_test_acc: {original_avg_test_acc}, "
              f"original: train_acc: {train_acc}, test_acc: {test_acc}, "
              f"{str(mutual_class[0])}_to_{str(mutual_class[0])}: {original_conf_matrix[mutual_class[0]][mutual_class[0]] + original_conf_matrix[mutual_class[1]][mutual_class[1]]}, "
              f"{str(mutual_class[1])}_to_{str(mutual_class[0])}: {original_conf_matrix[mutual_class[0]][mutual_class[1]] + original_conf_matrix[mutual_class[1]][mutual_class[0]]}")

    train_optimizer = optim.SGD(original_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    loss_function_ce = nn.CrossEntropyLoss()
    for epoch in range(1, 31):
        filters_for_aspect = None
        index = 0
        original_model_state_dict = model_state_dict_all[index]
        for mutual_class in mutual_classes:
            if index != 0:
                for current in filters_for_aspects:
                    if current['mutual_class'] == mutual_class:
                        filters_for_aspect = current['sensitive_filters_class_both']

                state_dict_temp = model_state_dict_all[index]
                for layer in original_model_state_dict:
                    layer_size = original_model_state_dict[layer].size()
                    if len(layer_size) == 4:
                        for filter_index in range(layer_size[0]):
                            if {'layer': layer, 'filter_index': filter_index} not in filters_for_aspect:
                                original_model_state_dict[layer][filter_index] = state_dict_temp[layer][filter_index]
            index += 1

        original_model.load_state_dict(original_model_state_dict)
        for index, (images, labels) in enumerate(train_loader):
            if model_args.gpu:
                images = images.to(settings.CUDA)
                labels = labels.to(settings.CUDA)
            train_optimizer.zero_grad()
            outputs = original_model(images)
            loss = loss_function_ce(outputs, labels)
            loss.backward()
            train_optimizer.step()

        train_acc = test(model=original_model, test_loader=train_loader)
        test_acc = test(model=original_model, test_loader=test_loader)
        avg_test_acc, test_acc_class = test_class(model_args=model_args, model=original_model, test_loader=test_loader)
        ground_truth_list, prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)
        conf_matrix = confusion_matrix(ground_truth_list, prediction_list)

        for mutual_class in mutual_classes:
            print(f"Epoch: {epoch}, mutual_class: {mutual_class}, epoch: {epoch}, avg_test_acc: {avg_test_acc}, "
                  f"train_acc: {train_acc}, test_acc: {test_acc}, "
                  f"{str(mutual_class[0])}_to_{str(mutual_class[0])}: {conf_matrix[mutual_class[0]][mutual_class[0]] + conf_matrix[mutual_class[1]][mutual_class[1]]}, "
                  f"{str(mutual_class[0])}_to_{str(mutual_class[1])}: {conf_matrix[mutual_class[0]][mutual_class[1]] + conf_matrix[mutual_class[1]][mutual_class[0]]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for data loader')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-num_class', type=float, default=100, help='class number')
    parser.add_argument('-dataset_name', type=str, default='cifar100', help='dataset name')
    parser.add_argument('-repairing', type=bool, default=False, help='repairing the original model')
    args = parser.parse_args()

    net = get_network(args)
    net.load_state_dict(torch.load(args.weights, map_location=settings.CUDA))
    n_classes = args.num_class

    # training data loader
    training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    # test data loader
    testing_loader = get_test_dataloader_all(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    # avg_train_acc, train_acc_class = test_class(net, training_loader)
    # print("Average training accuracy is: ", train_acc)
    # print("Test accuracy for each class: ", train_acc_class)

    # 47: maple_tree, 52: oak_tree; 35: girl, 98: woman; 13: bus, 81: street_car
    mutual_misclassified_classes = [[47, 52], [35, 98], [13, 81]]

    # find all the sensitive filters corresponding to the classes
    filters_in_all_aspects, filters_in_3_aspects = sensitive_filters(model_args=args, model=net, test_loader=testing_loader, mutual_classes=mutual_misclassified_classes, noise_var=0.1)
    for filters_in_1_aspects in filters_in_3_aspects:
        print(f"Mutual_class:{filters_in_1_aspects['mutual_class']}, Number of filters for both classes {len(filters_in_1_aspects['sensitive_filters_class_both'])}, "
              f"Number of filters for class {filters_in_1_aspects['mutual_class'][0]}: {len(filters_in_1_aspects['sensitive_filters_class_0'])}, "
              f"Number of filters for class {filters_in_1_aspects['mutual_class'][1]}: {len(filters_in_1_aspects['sensitive_filters_class_1'])}")

    net.load_state_dict(torch.load(args.weights, map_location=settings.CUDA))

    if args.repairing:
        model_state_dicts_3_aspects = controlled_training(model_args=args, model=net, train_loader=training_loader, test_loader=testing_loader, mutual_classes=mutual_misclassified_classes, filters_for_aspects=filters_in_3_aspects, noise_var=0.1)
    else:
        print("Model Repair with Incremental Patching")
        state_dict_3 = []
        state_dict_3.append(torch.load('results_mobilenet_cifar100/var_0.1/mutual_class_[47, 52]_noise_var_0.1_epoch_5.pth', map_location=settings.CUDA))
        state_dict_3.append(torch.load('results_mobilenet_cifar100/var_0.1/mutual_class_[35, 98]_noise_var_0.1_epoch_20.pth', map_location=settings.CUDA))
        state_dict_3.append(torch.load('results_mobilenet_cifar100/var_0.1/mutual_class_[13, 81]_noise_var_0.1_epoch_22.pth', map_location=settings.CUDA))
        patching(model_args=args, original_model=net, model_state_dict_all=state_dict_3, mutual_classes=mutual_misclassified_classes, filters_for_aspects=filters_in_3_aspects, train_loader=training_loader, test_loader=testing_loader)
