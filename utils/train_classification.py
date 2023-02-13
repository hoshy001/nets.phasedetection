import argparse
import os
import sys
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.profiler
import torch.utils.data

sys.path.append("..")

from model import feature_transform_regularizer, PCCT
from model import PhaseDataset, PointNetCls

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--batch_size', type=int, default=32, help='Input batch size')
    parser.add_argument(
        '-n', '--num_points', type=int, default=2000,
        help='Number of points sampled from the point cloud.')
    parser.add_argument(
        '-w', '--num_workers', type=int,
        help='Number of subprocesses to use for data loading', default=4)
    parser.add_argument(
        '-e', '--num_epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument(
        '-o', '--output_folder', type=str, default='cls', help='Output folder name')
    parser.add_argument(
        '-l', '--load_model', type=str, default='',
        help='A string towards a saved model path')
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, help='dataset path')
    parser.add_argument(
        '-f', '--feature_transform', action='store_true',
        help='use feature transform')
    parser.add_argument(
        '-c', '--save_critical_points', action='store_true',
        help='save critical points in numpy arrays')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        choices=('PointNet', 'PCCT'), help='Model name PointNet or PCCT')
    parser.add_argument(
        '-s', '--random_seed', type=int, default=random.randint(1, 10000),
        help='Input random seed')
    parser.add_argument(
        '-p', '--preprocess_data', action='store_true',
        help='preprocess the data')
    parser.add_argument(
        '-r', '--random_sample', action='store_true',
        help='use random sampling otherwise it uses FPS to select points from each point cloud')
    parser.add_argument(
        '-lr', type=float, default=1e-4,
        help='Adam algorithm learning rate.')

    opt = parser.parse_args()

    print(opt)

    def blue(s): return f'\033[91m{s}\033[0m'

    # seed everything
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)

    # choose dataset
    dataset = PhaseDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train',
        data_augmentation=True,
        preprocess_data=opt.preprocess_data,
        random_sample=opt.random_sample)

    test_dataset = PhaseDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False,
        preprocess_data=False,
        random_sample=True)

    # loading train & test data
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.num_workers),
        pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.num_workers))

    print(f'Number of train: {len(dataset)} & test: {len(test_dataset)} data')

    num_classes = len(dataset.classes)

    print(f'number of classes: {num_classes}')

    try:
        os.makedirs(opt.output_folder)
    except OSError:
        pass

    # setting up classifiers
    if opt.model == 'PointNet':
        classifier = PointNetCls(
            k=num_classes,
            feature_transform=opt.feature_transform)
    else:
        classifier = PCCT(
            k=num_classes,
            p=0.3,
            feature_transform=opt.feature_transform,
            blocks=4)

    if opt.load_model != '':
        classifier.load_state_dict(torch.load(opt.load_model))

    classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(),
                           lr=opt.lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_batch = len(dataset) / opt.batch_size

    # output test loss and accuracy
    test_out = (f'test_out_npts_{opt.num_points}.dat')
    ftest = open(test_out, 'w')
    ftest.write('num_epoch loss %s acc_overall\n' %
                (' '.join('class'+str(num) for num in range(num_classes))))
    train_out = (f'train_out_npts_{opt.num_points}.dat')
    ftrain = open(train_out, 'w')
    ftrain.write('num_epoch loss %s acc_overall\n' %
                 (' '.join('class' + str(num) for num in range(num_classes))))

    # train the model
    model_path = os.path.join(opt.output_folder, f'pts_{opt.num_points}')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    for epoch in range(opt.num_epochs):
        save_critical_points = opt.save_critical_points and epoch == opt.num_epochs - 1
        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(
                points, save_critical_points=save_critical_points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            # write out the train loss and accuracy
            acc_class = []
            for m in range(num_classes):
                ind_m = (target.data.cpu().numpy().ravel() == m)
                if np.sum(ind_m) == 0:
                    acc_m = 0
                else:
                    acc_m = np.sum((pred_choice.data.cpu().numpy().ravel()[
                                   ind_m] == m))/np.sum(ind_m)
                acc_class.append(acc_m)

            print(
                f"[{epoch}: {i}/{int(num_batch)}] train loss: {loss.item()} accuracy: {correct.item()/float(opt.batch_size):.3f}")
            ftrain.write(
                f"{epoch} {loss.item()} {correct.item() / float(opt.batch_size)} \n")

            if i % (num_batch+1) == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()

                # write out the test loss and accuracy
                acc_class = []
                for m in range(num_classes):
                    ind_m = (target.data.cpu().numpy().ravel() == m)
                    if np.sum(ind_m) == 0:
                        acc_m = 0
                    else:
                        acc_m = np.sum((pred_choice.data.cpu().numpy().ravel()[
                                       ind_m] == m))/np.sum(ind_m)
                    acc_class.append(acc_m)

                print(
                    f"[{epoch}: {i}/{int(num_batch)}] {blue('test')} loss: {loss.item()} accuracy: {correct.item()/float(opt.batch_size):.3f}")
                ftest.write(
                    f"{epoch} {loss.item()} {correct.item() / float(opt.batch_size)} \n")
        scheduler.step()
        torch.save(classifier.state_dict(), '%s/pts_%s/cls_model_%d.pth' %
                   (opt.output_folder, str(opt.num_points), epoch))

    # Close the logs for train and test
    ftest.close()
    ftrain.close()

    # testing
    # Initialization for testing the trained model
    total_correct = 0
    total_testset = 0
    acc_classes = []
    conf_mats = []

    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        acc_class = []
        conf_mat = np.zeros((num_classes, num_classes))
        for n in range(num_classes):
            for m in range(num_classes):
                ind_m = (target.data.cpu().numpy().ravel() == m)
                if np.sum(ind_m) != 0:
                    conf_mat[m, n] = np.sum(
                        (pred_choice.data.cpu().numpy().ravel()[ind_m] == n))

                # Record the diagonal elements to calculate accuracy
                if m == n:
                    if np.sum(ind_m) == 0:
                        acc_m = 0
                    else:
                        acc_m = np.sum((pred_choice.data.cpu().numpy().ravel()[
                                       ind_m] == m)) / np.sum(ind_m)
                    acc_class.append(acc_m)

        acc_classes.append(acc_class)
        conf_mats.append(conf_mat)
        total_correct += correct.item()
        total_testset += points.size()[0]

    final_acc = total_correct / float(total_testset)

    ave_conf_mats = np.mean(conf_mats, 0)
    print("Final accuracy {}".format(final_acc))

    # Save the confusion matrix for computing accuracy, precisions, recalls, and f1 scores.
    np.savetxt(("conf_mat_%d.dat" % opt.num_points),
               np.sum(conf_mats, 0), fmt='% 4d', delimiter=',')
