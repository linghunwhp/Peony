import argparse
import torch
from conf import settings
from utils import get_network, get_test_dataloader_all, test_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
torch.set_printoptions(profile="full")
import xlwt


def plot_confusion_matrix(model_args, y_true, y_pred, image_name='confusion_matrix.png', title='Confusion Matrix'):
    classes = list(range(model_args.num_class))
    cm = confusion_matrix(y_true, y_pred)       # get confusion matrix
    plt.figure(figsize=(20, 10), dpi=100)
    np.set_printoptions(precision=2)
    ind_array = np.arange(len(classes))
    print(ind_array.shape)
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    x_locations = np.array(range(len(classes)))
    plt.xticks(x_locations, classes, rotation=90)
    plt.yticks(x_locations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(image_name, format='png')
    plt.show()


def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1


def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')


def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    plt.show()


def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges", fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize=[9,9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for data loader')
    parser.add_argument('-num_class', type=float, default=100, help='class number')
    parser.add_argument('-dataset_name', type=str, default='cifar100', help='dataset name')
    args = parser.parse_args()

    net = get_network(args)
    net.load_state_dict(torch.load(args.weights, map_location=settings.CUDA))
    n_classes = args.num_class

    # # training accuracy on train data
    testing_loader = get_test_dataloader_all(
        # settings.TINY_IMAGENET_MEAN,
        # settings.TINY_IMAGENET_STD,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.dataset_name
    )

    ground_truth_list, prediction_list = get_ground_truth_prediction(model_args=args, model=net, data_loader=testing_loader)

    # first method
    # plot_confusion_matrix(model_args=args, y_true=ground_truth_list, y_pred=prediction_list)

    # # second method
    # columns = list(range(args.num_class))
    # annot = True;
    # cmap = 'Oranges';
    # fmt = '.2f'
    # lw = 0.5
    # cbar = False
    # show_null_values = 2
    # pred_val_axis = 'y'
    # fz = 12;
    # figsize = [90,90];
    #
    # if(len(ground_truth_list) > 10):
    #     fz=9;
    #     figsize=[14, 14];
    #
    # plot_confusion_matrix_from_data(ground_truth_list, prediction_list, columns, annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

    loss_function_ce = torch.nn.CrossEntropyLoss()
    test_acc, _ = test_model(model_args=args, model=net, test_loader=testing_loader, loss_function=loss_function_ce)
    print(f"Test Acc is {test_acc}")

    conf_matrix = confusion_matrix(ground_truth_list, prediction_list)
    cls_list = []
    correct_value_list = []
    # misclassification
    mis_cls_list = []
    mis_cls_value_list = []
    mis_cls_value_from_list = []
    mis_cls_value_to_list = []

    # # original model
    # for cls in range(args.num_class):
    #     misclassification = conf_matrix[cls]
    #     value, index = torch.tensor(misclassification).topk(2, 0, largest=True, sorted=True)
    #     if (value[1] + conf_matrix[:, cls][int(index[1])]) > 10:
    #         cls_list.append(cls)
    #         correct_value_list.append(int(value[0]))
    #         mis_cls_list.append(int(index[1]))
    #         mis_cls_value_list.append(int(value[1])+int(conf_matrix[:, cls][int(index[1])]))
    #         mis_cls_value_from_list.append(int(conf_matrix[cls][int(index[1])]))
    #         mis_cls_value_to_list.append(int(conf_matrix[:, cls][int(index[1])]))

    # 200 class
    # # original model
    # for cls in range(args.num_class):
    #     misclassification = conf_matrix[cls]
    #     value, index = torch.tensor(misclassification).topk(2, 0, largest=True, sorted=True)
    #     if (value[1] + conf_matrix[:, cls][int(index[1])])*2 > 10:
    #         cls_list.append(cls)
    #         correct_value_list.append(int(value[0]))
    #         mis_cls_list.append(int(index[1]))
    #         mis_cls_value_list.append(int(value[1])+int(conf_matrix[:, cls][int(index[1])]))
    #         mis_cls_value_from_list.append(int(conf_matrix[cls][int(index[1])]))
    #         mis_cls_value_to_list.append(int(conf_matrix[:, cls][int(index[1])]))

    # fixed model
    # mobilenet-cifar100
    # need_class = list(range(100))
    # mis_to_class = [57, 62, 35, 19, 74, 25, 14, 24, 48, 57, 61, 35, 33, 81, 62, 31, 9, 37, 6, 38, 25, 46, 61, 71, 7, 20, 45, 18, 10, 34, 73, 19, 67, 96, 38, 11, 65, 81, 97, 23, 99, 89, 88, 88, 27, 44, 98, 52, 8, 60, 74, 78, 47, 57, 92, 4, 59, 83, 13, 52, 71, 10, 70, 74, 50, 38, 97, 73, 12, 40, 92, 60, 55, 30, 50, 65, 23, 55, 99, 6, 97, 13, 62, 53, 25, 58, 40, 94, 34, 8, 81, 73, 62, 73, 5, 73, 52, 64, 35, 78]
    # need_class = [2, 5, 10, 11, 13, 20, 22, 23, 25, 28, 30, 32, 33, 35, 44, 46, 47, 50, 52, 55, 57, 58, 59, 60, 61, 62, 67, 70, 71, 72, 73, 74, 78, 81, 83, 90, 92, 95, 96, 98, 99]
    # mis_to_class = [35, 25, 61, 35, 81, 25, 61, 71, 20, 10, 73, 67, 96, 11, 27, 98, 52, 74, 47, 4, 83, 13, 52, 71, 10, 70, 73, 92, 60, 55, 30, 50, 99, 13, 53, 81, 62, 73, 52, 35, 78]

    # mobilenet-tiny
    # need_class = [2, 3, 14, 16, 17, 18, 19, 27, 29, 30, 32, 34, 37, 38, 39, 40, 41, 42, 46, 48, 49, 50, 61, 62, 68, 70, 72, 73, 74, 82, 85, 86, 93, 94, 96, 101, 104, 114, 116, 117, 120, 121, 126, 128, 134, 135, 153, 154, 155, 157, 164, 171, 172, 174, 177, 178, 179, 182, 186, 187, 190, 191, 194, 195, 196, 197, 198]
    # mis_to_class = [3, 2, 196, 10, 46, 19, 18, 26, 27, 32, 30, 33, 38, 37, 42, 42, 37, 39, 196, 50, 50, 49, 126, 120, 171, 94, 172, 135, 158, 93, 105, 128, 82, 70, 197, 132, 174, 164, 70, 155, 62, 70, 61, 86, 62, 73, 94, 157, 117, 154, 114, 68, 72, 104, 182, 183, 182, 177, 187, 188, 182, 192, 195, 194, 46, 198, 197]

    # # shufflenetv2-cifar100
    need_class = [2, 3, 4, 5, 10, 11, 13, 22, 23, 25, 26, 27, 28, 30, 32, 33, 35, 44, 46, 47, 50, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 67, 71, 72, 73, 74, 78, 80, 81, 83, 84, 90, 92, 95, 96, 98, 99]
    mis_to_class = [35, 4, 74, 25, 61, 46, 81, 61, 71, 5, 45, 44, 10, 73, 67, 96, 98, 78, 98, 52, 74, 47, 92, 72, 83, 13, 52, 71, 10, 92, 74, 50, 73, 23, 55, 67, 50, 99, 38, 90, 53, 5, 81, 62, 73, 47, 46, 78]

    # # shufflenetv2-tiny
    # need_class = [2, 3, 14, 24, 26, 27, 28, 30, 32, 33, 37, 39, 40, 42, 43, 47, 50, 61, 68, 70, 73, 80, 84, 85, 86, 91, 94, 101, 104, 116, 117, 121, 126, 127, 128, 134, 135, 137, 138, 148, 153, 155, 160, 164, 174, 177, 180, 182, 186, 187, 188, 190, 191, 192, 196, 197, 198]
    # mis_to_class = [3, 2, 196, 28, 27, 26, 24, 32, 30, 34, 38, 42, 42, 40, 39, 31, 49, 126, 171, 94, 135, 172, 137, 134, 128, 195, 70, 132, 174, 70, 155, 133, 61, 128, 127, 85, 73, 84, 100, 196, 94, 117, 172, 114, 104, 190, 188, 190, 187, 188, 180, 177, 190, 182, 46, 198, 197]

    index = 0
    for cls in need_class:
        cls_list.append(cls)
        correct_value_list.append(int(conf_matrix[cls][cls]))
        mis_cls_list.append(mis_to_class[index])
        mis_cls_value_list.append(int(conf_matrix[cls][mis_to_class[index]]+conf_matrix[:, cls][mis_to_class[index]]))
        mis_cls_value_from_list.append(int(conf_matrix[cls][mis_to_class[index]]))
        mis_cls_value_to_list.append(int(conf_matrix[:, cls][mis_to_class[index]]))
        index += 1

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('sheet 1')
    row0 = ["mutual mis class", "class", "correct percent", "mutual mis class", "mutual misclassification percentage", "from", "to"]
    for i in range(len(row0)):
        sheet1.write(0, i, row0[i])
    for i in range(1, len(cls_list)+1):
        sheet1.write(i, 0, str(cls_list[i-1])+'/'+str(mis_cls_list[i-1]))
        sheet1.write(i, 1, cls_list[i-1])
        # sheet1.write(i, 2, correct_value_list[i-1]*2)
        # sheet1.write(i, 3, mis_cls_list[i-1])
        # sheet1.write(i, 4, mis_cls_value_list[i-1]*2)
        # sheet1.write(i, 5, mis_cls_value_from_list[i-1]*2)
        # sheet1.write(i, 6, mis_cls_value_to_list[i-1]*2)
        sheet1.write(i, 2, correct_value_list[i-1])
        sheet1.write(i, 3, mis_cls_list[i-1])
        sheet1.write(i, 4, mis_cls_value_list[i-1])
        sheet1.write(i, 5, mis_cls_value_from_list[i-1])
        sheet1.write(i, 6, mis_cls_value_to_list[i-1])
    f.save(r'D:\data_analysis_4_fixed_model.xls')

    print(cls_list)
    print(correct_value_list)
    print(mis_cls_list)
    print(mis_cls_value_list)
    print(mis_cls_value_from_list)
    print(mis_cls_value_to_list)

