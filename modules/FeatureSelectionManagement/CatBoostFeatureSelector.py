import pickle
import sys
import os
import numpy as np
import math
sys.path.insert(0, os.getcwd())
import parameters as para

class CatBoostFeatureSelector:
    def get_selector():
        sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
        import modules.ClassifiersManagement.helper as classifierHelper
        import catboost as cb
        # import pandas as pd
        try:
            clf = pickle.load(open(classifierHelper.get_special_name(
                'pickled', 'CatBoostFeatureSelector', '.pickle',with_featureSelection_prefix=False,with_randomizeSearch_prefix=False), "rb"))
        except:
            sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
            from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array
            feature_emotion_X_Y_array = extract_feature_emotion_X_y_array(filter=False)
            X = feature_emotion_X_Y_array['X']
            y = feature_emotion_X_Y_array['y']

            # Create a CatBoostClassifier model
            clf = cb.CatBoostClassifier(
                                        depth=4,
                                        )
            # Fit the model to the data
            clf.fit(X, y)
            pickle.dump(clf, open(classifierHelper.get_special_name(
                'pickled', 'CatBoostFeatureSelector', '.pickle',with_featureSelection_prefix=False,with_randomizeSearch_prefix=False), "wb"))
        return clf
     
    def get_feature_importance():
        model = CatBoostFeatureSelector.get_selector()
        feature_importance = model.get_feature_importance()
        return feature_importance

    def get_boolean_array_for_most_important_feature(ratio=para.selection_ratio):
        feature_importance = CatBoostFeatureSelector.get_feature_importance()
        # print('hi',ratio)
        # exit()
        index = int((len(feature_importance)-1)*ratio)
        greater_than = np.sort(feature_importance)[::-1][index]
        # print(ratio)
        # print(feature_importance >= greater_than)
        # exit()
        return feature_importance >= greater_than
        # exit()
        return feature_importance >= math.sqrt(max(feature_importance)*np.average(feature_importance))*ratio
        return feature_importance >= max(feature_importance)*ratio
        return feature_importance > np.average(feature_importance)*ratio
        # return feature_importance >= np.min(feature_importance)
    
    def filter_features(X,ratio=para.selection_ratio):
        import numpy as np
        # Convert the input list to a numpy array
        X = np.array(X)
        y = CatBoostFeatureSelector.get_boolean_array_for_most_important_feature(ratio=para.selection_ratio)
        return X[:,y]

if __name__ == '__main__':
    sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
    import modules.ClassifiersManagement.helper as classifierHelper
    import catboost as cb
    sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
    from modules.FeaturesManagement.main import extract_feature_emotion_X_y_array

    feature_emotion_X_Y_array = extract_feature_emotion_X_y_array(filter=False)
    X = feature_emotion_X_Y_array['X']
    y = feature_emotion_X_Y_array['y']
    # Create a CatBoostClassifier model
    clf = cb.CatBoostClassifier()

    # Fit the model to the data
    clf.fit(X, y)
    pickle.dump(clf, open(classifierHelper.get_special_name('pickled', 'CatBoostFeatureSelector', '.pickle',with_featureSelection_prefix=False,with_randomizeSearch_prefix=False), "wb"))