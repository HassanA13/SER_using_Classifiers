# from sklearn.ensemble import GradientBoostingClassifier as Classifier
# import pickle
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array
sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
import modules.ClassifiersManagement.helper as classifierHelper
# from modules.ClassifiersManagement.AdaBoostClassifier import AdaBoostClassifier as Classifier
# from sklearn.ensemble import GradientBoostingClassifier as Classifier
# from modules.ClassifiersManagement.GradientBoostingClassifier import GradientBoostingClassifier as Classifier
# from modules.ClassifiersManagement.BaggingClassifier import BaggingClassifier as Classifier
# from modules.ClassifiersManagement.RandomForestClassifier import RandomForestClassifier as Classifier


def swap(x, y):
    return y, x


def test(para_datasets, Classifier, test_size=1, with_show=True):
    if para.with_random_search:
        clf = Classifier.get_classifier_through_randomized_search_cv()
    else:
        clf = Classifier.get_classifier()

    para_selection_ratio = 1
    para.selection_ratio, para_selection_ratio = swap(para.selection_ratio, para_selection_ratio)
    para.datasets, para_datasets = swap(para.datasets, para_datasets)

    # print(para.selection_ratio)
    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array()
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']


    para.selection_ratio, para_selection_ratio = swap(para.selection_ratio, para_selection_ratio)
    para.datasets, para_datasets = swap(para.datasets, para_datasets)

    sys.path.insert(0, os.getcwd()+"\modules\FeatureSelectionManagement")
    from modules.FeatureSelectionManagement.CatBoostFeatureSelector import CatBoostFeatureSelector
    print(f"The Shape of feature before Filtering :{X.shape}")
    # print(para.selection_ratio)
    X = CatBoostFeatureSelector.filter_features(X,ratio=para.selection_ratio)
    # print(para.selection_ratio)
    print(f"The Shape of feature after Filtering :{X.shape}")
    # print(para.selection_ratio)
    # exit()

    para.datasets, para_datasets = swap(para.datasets, para_datasets)

    if test_size != 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    else:
        X_test = X
        y_test = y

    y_prediction = clf.predict(X_test)
    # if para.with_random_search:
    #     classifierHelper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name=f'{Classifier.__name__}',
    #                                       title=f"Accuracy: {clf.score(X_test, y_test)} - RandomizeSearch_{Classifier.__name__}\n{classifierHelper.get_special_name(folder_name='',prefix='')}\n{clf.best_params_}")
    # else:
    classifierHelper.confusion_matrix(y_test=y_test, y_prediction=y_prediction, classifier_name=f'{Classifier.__name__}',title=f"Accuracy: {clf.score(X_test, y_test)} - {Classifier.__name__}\n{classifierHelper.get_special_name(folder_name='',prefix='')}", normalize='true', with_show=with_show)

    para.datasets, para_datasets = swap(para.datasets, para_datasets)


if __name__ == '__main__':
    from modules.ClassifiersManagement.AdaBoostClassifier import AdaBoostClassifier
    from modules.ClassifiersManagement.BaggingClassifier import BaggingClassifier 
    from modules.ClassifiersManagement.GradientBoostingClassifier import GradientBoostingClassifier
    from modules.ClassifiersManagement.KNeighborsClassifier import KNeighborsClassifier
    from modules.ClassifiersManagement.MLPClassifier import MLPClassifier
    from modules.ClassifiersManagement.RandomForestClassifier import RandomForestClassifier
    from modules.ClassifiersManagement.SVCClassifier import SVCClassifier
    classifiers = [
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    # KNeighborsClassifier,
    # MLPClassifier,
    RandomForestClassifier,
    # SVCClassifier,
    ]
    features = [
        'MFCC',
        'SSC',
        'Chroma',
        'MelSpectrogram',
        'Contrast',
        'Tonnetz',
        'F0',
    ]
    selection_ratios = [
        # 0.1,
        # 0.2,
        0.3,
        # 0.4,
        # # 1,
        # 0.9,
        # 0.8,
        # 0.7,
        # 0.6,
        # 0.5,
    ]
    with_random_searchs = [
        True,
        # False
    ]
    for selection_ratio in selection_ratios:
        for Classifier in classifiers:
            for with_random_search in with_random_searchs:
                for Feature in features:
                    para.with_random_search = with_random_search
                    para.selection_ratio = selection_ratio
                    para.features = [Feature]
                    # test([
                    #     # 'custom_arabic',
                    #     # 'TESS',
                    #     # 'emo-db',
                    #     'tess_ravdess',
                    #     # 'AudioWAV',
                    # ], Classifier=Classifier, test_size=para.test_size, with_show=False)

                    test([
                        # 'PD',
                        # 'CA',
                        # 'TESS',
                        # 'emo-db',
                        'tess_ravdess',
                        # 'AudioWAV',
                    ], Classifier=Classifier, with_show=False)
