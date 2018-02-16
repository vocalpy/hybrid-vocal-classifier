import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def confusion_matrix(cm, classes,
                     normalize=False,
                     title='Confusion matrix',
                     cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Adapted from from: 
    http://scikit-learn.org/stable/auto_examples/model_selection/
    plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#    adapted from:
#    http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def grid_search(scores, gamma_range, C_range):
    """plot results from grid search

    adapted from:
    http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    """
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


def learning_curve(train_sample_sizes, test_metric_vals, test_lbl='test',
                   train_metric_vals=None, train_lbl='train',
                   figsize=(10,5)):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sample_sizes,
            test_metric_vals,
            label=test_lbl,
            linestyle=':',
            marker='o')
    if train_metric_vals:
        ax.plot(train_sample_sizes,
                train_metric_vals,
                label=train_lbl,
                linestyle=':',
                marker='o')
    plt.legend()
    plt.xticks(train_sample_sizes)
    plt.show()
