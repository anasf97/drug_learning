import numpy as np
from scipy.spatial import distance
from collections import Counter

class ApplicabilityDomain():

    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def fit(self):
        distances = np.array([distance.cdist([x], self.x_train) for x in self.x_train])
        distances_sorted = [np.sort(d[0]) for d in distances]
        d_no_ii = [ d[1:] for d in distances_sorted]
        k = int(round(pow(len(self.x_train), 1/3)))

        d_means = [np.mean(d[:k][0]) for d in d_no_ii] #medium values
        Q1 = np.quantile(d_means, .25)
        Q3 = np.quantile(d_means, .75)
        IQR = Q3 - Q1
        d_ref = Q3 + 1.5*(Q3-Q1) #setting the reference value
        n_allowed =  []
        all_allowed = []
        for i in d_no_ii:
            d_allowed = [d for d in i if d <= d_ref]
            all_allowed.append(d_allowed)
            n_allowed.append(len(d_allowed))

        #selecting minimum value not 0:
        min_val = [np.sort(n_allowed)[i] for i in range(len(n_allowed)) if np.sort(n_allowed)[i] != 0]

        #replacing 0's with the min val
        n_allowed = [n if n!= 0 else min_val[0] for n in n_allowed]
        c = Counter(n_allowed)
        print(c)
        all_d = [sum(all_allowed[i]) for i, d in enumerate(d_no_ii)]
        self.thresholds = np.divide(all_d, n_allowed) #threshold computation
        self.thresholds[np.isinf(self.thresholds)] = min(self.thresholds) #setting to the minimum value where infinity
        return self.thresholds

    def predict(self):
        test_names= ["sample_{}".format(i) for i in range(self.x_test.shape[0])]
        d_train_test = np.array([distance.cdist([x], self.x_train) for x in self.x_test])
        count_active = []
        count_inactive = []

        for i, name in zip(d_train_test, test_names): # for each sample
            idxs = [j for j,d in enumerate(i[0]) if d <= self.thresholds[j]] #saving indexes of training with threshold < distance
            count_active.append(len([self.y_train.tolist()[i] for i in idxs if self.y_train[i] == 1]))
            count_inactive.append(len([self.y_train.tolist()[i] for i in idxs if self.y_train[i] == 0]))
        self.n_insiders = np.array(count_active) + np.array(count_inactive)
        return self.n_insiders
