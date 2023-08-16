import time
import os
import sys
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.Feature import Feature

def check_directory(dir_path):
        # Specify the directory path
    # dir_path = "/path/to/directory"
    try:
        # Check if the directory exists
        if not os.path.exists(dir_path):
            # Create the directory if it does not exist
            os.makedirs(dir_path)
            # print(f"Directory {dir_path} created!")
        # else:
            # print(f"Directory {dir_path} already exists!")
    except:
        pass

def get_special_name(folder_name='memory', prefix='', extension='',with_featureSelection_prefix=True,with_randomizeSearch_prefix=True):
    
    if with_featureSelection_prefix:
        name = f'with_select_ratio_{para.selection_ratio}_'
    else:
        name = ''
    name = name + prefix
    for dataset in para.datasets:
        name = name + \
            f"_{dataset}"
    for feature in para.features:
        name = name + \
            f"_{feature}"
    for emotion in para.emotions:
        name = name + \
            f"_{emotion}"

    if para.with_random_search and with_randomizeSearch_prefix:
        folder_name = folder_name + "\\randomizeSearch"

    check_directory(folder_name)
    
    name = os.path.join(folder_name, f"{name}{extension}")
    # print(name)
    # exit()
    return name


def confusion_matrix(y_test, y_prediction, classifier_name, title='', normalize='true',with_show=True):
    if para.with_random_search:
        title = title + "\n{clf.best_params_}"
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    indices = list(set(y_prediction).union(y_test))
    indices = [int(x) for x in indices]

    class_names = [para.emotions[i] for i in indices]

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_prediction,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )

    disp.ax_.set_title(title)

    fig = plt.gcf()

    if with_show:
        plt.show()
    
    fig.savefig(get_special_name(folder_name=f'statistics\\{classifier_name}', prefix=f'_{classifier_name}', extension='.png'), bbox_inches='tight')


def predict(path, classifier):
    feature = Feature(path).extract_feature().reshape(1, -1)

    sys.path.insert(0, os.getcwd()+"\modules\FeatureSelectionManagement")
    from  modules.FeatureSelectionManagement.CatBoostFeatureSelector import CatBoostFeatureSelector 
    feature = CatBoostFeatureSelector.filter_features(feature)

    return para.emotions[int(classifier.predict(feature)[0])]
