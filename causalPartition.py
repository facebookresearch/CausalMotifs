import numpy as np
import pandas as pd
import statsmodels.api as sm
import gc
import operator
import networkx as nx


class causalPartition:
    df = None # the whole dataset
    probabilities = None # the Monte Carlo probabilities, a dict, each element represents a dimension of the intervention vector
    # each element is a matrix [num_nodes * num_bootstrap]
    result_separate = None
    result_eht = None
    treatment = None # the name of treatment feature (should belong to the dict probabilities)
    df_train = None
    df_est = None

    def __init__(self, df, probabilities, treatment, ratio_train=0.5):
        """
        Ratio_train is the ratio in the training set, which is used to split the sample in the beginning to help construct honest estimator.
        By default it is 50% vs 50% to make the training and estimation sets have the roguhly same widths for CIs_contain_zero
        """
        self.df = df
        self.probabilities = probabilities
        self.treatment = treatment
        self.idx_tr = np.random.random(len(df)) < ratio_train  # sample the training set
        self.idx_est = np.logical_not(self.idx_tr)  # sample the estimation set
        self.df_train = df[self.idx_tr]
        self.df_est = df[np.logical_not(self.idx_tr)]
        self.result_separate = None
        self.est_result_separate_eht = None

    # for each observation, if there is small probability of belong to the partition defined by rules 
    def _contain_zero(self, probabilities, rules, eps, delta, treated=None):
        """
        For each observation (indicated by an element in the vector),
        whether it has <= eps probability to belong to the partition implied by [rules]
        Treated: == 1/0 if we want to append the rule for the treatment variable
                 == none otherwise
        """
        if treated is None:
            return np.mean(np.product([self.probabilities[key] <= th for key, sign, th in rules if sign == 0] + \
                            [self.probabilities[key] > th for key, sign, th in rules if sign == 1],
                           axis=0) > 0, axis=1
                          ) <= eps

        else:
            # Also consider the treated conditions for egos. 
            # In separate trees, the treatment conditions for egos should also be considered
            return np.mean(np.product([probabilities[key] <= th for key, sign, th in rules if sign == 0] + \
                            [probabilities[key] > th for key, sign, th in rules if sign == 1] + \
                             probabilities[self.treatment] == treated,
                           axis=0) > 0, axis=1
                          ) <= eps

    def _hajek_se(self, d, p, outcome):
        """
        - The taylor linearization for hajek se 
        - WLS is directly used for non-separate cases; but in this case S.E. is usually overestimated
        """
        average_hajek_var_up = np.sum( ((d[outcome]/p) ** 2) * (1 - p) )  # numerator
        average_hajek_var_down = np.sum( ((1.0/p) ** 2) * (1 - p) )  # denominator
        average_hajek_cov = np.sum( ((1.0/p) ** 2) * d[outcome] * (1 - p) )
        average_hajek_sum_up = np.sum(d[outcome]/p)  # numerator
        average_hajek_sum_down = np.sum(1.0/p)  # denominator

        se = np.sqrt(1.0 / (average_hajek_sum_down**2) * average_hajek_var_up + \
                (average_hajek_sum_up**2) / (average_hajek_sum_down**4) * average_hajek_var_down + \
                - 2.0 * average_hajek_sum_up / (average_hajek_sum_down**3) * average_hajek_cov)
        # Taylor linearization ((Sarndal, Swensson and Wretman, 1992, pp. 172-174)
        return se

    def _plot_tree(self, est_result_separate, node_id, prefix):
        if node_id > 1 and node_id % 2 == 0:
            sign = '<='
        elif node_id > 1 and node_id % 2 == 1:
            sign = '> '
        else:
            sign = ''
        if 'left_result' in est_result_separate:
            print('%s%s(%d) split %s at %f, n=%d, avg=%f, se=%f' % (prefix, sign, node_id, est_result_separate['feature'], 
                est_result_separate['threshold'], est_result_separate['N'], est_result_separate['hajek'], est_result_separate['hajek_se']))
            self._plot_tree(est_result_separate['left_result'], node_id*2, prefix+'\t')
            self._plot_tree(est_result_separate['right_result'], node_id*2+1, prefix+'\t')
        else:
            print('%s%s(%d) terminate, n=%d, avg=%f, se=%f' % (prefix, sign, node_id, 
                        est_result_separate['N'], est_result_separate['hajek'], est_result_separate['hajek_se']))
    
    def plot_tree(self, result=None):
        if 1 in result:
            print('treated')
            self._plot_tree(result[1], 1, '')
            print('non-treated')
            self._plot_tree(result[0], 1, '')
        else:
            self._plot_tree(result, 1, '')

    def _split_exposure_hajek(self, node_id, df, probabilities, feature_set, max_attempt, eps, delta, 
                              outcome, rules, N, current_mse, criteria={'non_trivial_reduction': 0},
                             first_split_treatment=True):
        """
        the actual splitting implementation for separate tree; 
        by recursion
        """
        b_feature = ''
        b_threshold = 0
        b_left = None
        b_right = None
        b_average_left_hajek = 0
        b_average_right_hajek = 0
        b_mse = 10000000000.0  # a very large mse

        ranges = {}
        # enumerate each feature
        for feature in feature_set:
            gc.collect()
            # find a more compact region
            upper = 1.
            lower = 0.
            for rule in rules:
                # rules: list of tuples to describe the decision rules
                # tuples(feature, 0/1: lower or upper bound, value)
                if rule[0] == feature:
                    if rule[1] == 0:
                        lower = np.maximum(rule[2], lower)
                    else:
                        upper = np.minimum(rule[2], upper)
            if lower >= upper:
                continue
            
            for k in range(max_attempt):
                if first_split_treatment and node_id == 1:
                    if feature != self.treatment or k != 0:
                        continue
                
                threshold = np.random.uniform(lower, upper)  # randomly select a threshold, left < , right >
                # make sure it is a valid split --- each observation should have non-trial (>eps) probability to belong to each partition
                cz_l = self._contain_zero(probabilities, rules+[(feature, 0, threshold)], eps, delta)
                cz_r = self._contain_zero(probabilities, rules+[(feature, 1, threshold)], eps, delta)
                if np.mean(cz_l) > delta or np.mean(cz_r) > delta:
                    continue
                    # if (almost) positivity can't be satisfied
                    
                idxs_left = np.product([df[key] <= th for key, sign, th in rules if sign == 0] + \
                       [df[key] > th for key, sign, th in rules if sign == 1] + \
                        [df[feature] <= threshold],
                       axis=0) > 0

                idxs_right = np.product([df[key] <= th for key, sign, th in rules if sign == 0] + \
                       [df[key] > th for key, sign, th in rules if sign == 1] + \
                        [df[feature] > threshold],
                       axis=0) > 0

                left = df[idxs_left]
                right = df[idxs_right]
                
                # generalized propensity score (probability of belonging in an exposure condition)
                propensities_left = np.mean(np.product([probabilities[key][idxs_left] <= th for key, sign, th in rules if sign == 0] + \
                           [probabilities[key][idxs_left] > th for key, sign, th in rules if sign == 1] + \
                            [probabilities[feature][idxs_left] <= threshold],
                           axis=0) > 0, axis=1)

                # generalized propensity score (probability of belonging in an exposure condition)
                propensities_right = np.mean(np.product([probabilities[key][idxs_right] <= th for key, sign, th in rules if sign == 0] + \
                           [probabilities[key][idxs_right] > th for key, sign, th in rules if sign == 1] + \
                            [probabilities[feature][idxs_right] > threshold],
                           axis=0) > 0, axis=1)
                # again, filter small propensities data points (usually should not filter or filter very few)
                
                if len(left) == 0 or len(right) == 0:
                    continue
                
                filter_left = propensities_left > 0
                left = left[filter_left]
                propensities_left = propensities_left[filter_left]
                
                filter_right = propensities_right > 0
                right = right[filter_right]
                propensities_right = propensities_right[filter_right]
                
                mod_left = sm.WLS(left[outcome], np.ones(len(left)), weights=1.0 / propensities_left)
                mod_right = sm.WLS(right[outcome], np.ones(len(right)), weights=1.0 / propensities_right)
                
                res_left = mod_left.fit()
                res_right = mod_right.fit()
                
                average_left_hajek = res_left.params[0] 
                average_right_hajek = res_right.params[0]

                average_left_hajek_se = self._hajek_se(left, propensities_left, outcome)
                average_right_hajek_se = self._hajek_se(right, propensities_right, outcome)

                mse_left = np.sum((1.0 / propensities_left) * ((res_left.resid) ** 2))
                mse_right = np.sum((1.0 / propensities_right) * ((res_right.resid) ** 2))
                mse = mse_left * len(left)/(len(left)+len(right)) + mse_right * len(right)/(len(left)+len(right))
                
                if mse < b_mse:
                    flag = True
                    assert len(criteria) > 0
                    if 'non_trivial_reduction' in criteria:
                        if not (mse < current_mse - criteria['non_trivial_reduction']):
                            flag = False
                    if 'reasonable_propensity' in criteria:
                        if not (np.abs(np.sum(1.0 / propensities_left)/len(df) - 1.0) <= criteria['reasonable_propensity'] \
                                and \
                                np.abs(np.sum(1.0 / propensities_right)/len(df) - 1.0) <= criteria['reasonable_propensity'] \
                               ):
                            flag = False
                    if 'separate_reduction' in criteria:
                        if not (mse_left < current_mse and mse_right < current_mse):
                            flag = False
                    if 'min_leaf_size' in criteria:
                        if not (len(left) >= criteria['min_leaf_size'] and len(right) >= criteria['min_leaf_size']):
                            flag = False
                    if flag:
                        b_feature = feature
                        b_mse = mse
                        b_mse_left = mse_left
                        b_mse_right = mse_right
                        b_threshold = threshold
                        b_average_left_hajek = average_left_hajek
                        b_average_right_hajek = average_right_hajek
                        b_average_left_hajek_se = average_left_hajek_se
                        b_average_right_hajek_se = average_right_hajek_se
                        b_left_den = np.sum(1.0 / propensities_left)
                        b_right_den = np.sum(1.0 / propensities_right)
                        b_left = left
                        b_right = right
                        b_left_rules = rules + [(feature, 0, threshold)]
                        b_right_rules = rules + [(feature, 1, threshold)]

        result = {}
        if b_feature != '':
            # if find a valid partition
            result_left = self._split_exposure_hajek(node_id*2, df, probabilities, feature_set, max_attempt, eps, delta, 
                                                     outcome, b_left_rules, len(b_left), b_mse_left, criteria)
            result_right = self._split_exposure_hajek(node_id*2+1, df, probabilities, feature_set, max_attempt, eps, delta, 
                                                      outcome, b_right_rules, len(b_right), b_mse_right, criteria)
            result['mse'] = result_left['mse'] * 1.0 * len(b_left)/(len(b_left)+len(b_right)) + \
                        result_right['mse'] * 1.0 * len(b_right)/(len(b_left)+len(b_right))
            result['feature'] = b_feature
            result['threshold'] = b_threshold
            result_left['hajek'] = b_average_left_hajek
            result_right['hajek'] = b_average_right_hajek
            result_left['hajek_se'] = b_average_left_hajek_se
            result_right['hajek_se'] = b_average_right_hajek_se
            result_left['N'] = len(b_left)
            result_right['N'] = len(b_right)
            result_left['den'] = b_left_den
            result_right['den'] = b_right_den
            result['left_result'] = result_left
            result['right_result'] = result_right
            return result
        else:
            result['mse'] = current_mse
            return result
        
    def _split_exposure_validate_eht(self, node_id, df_est, result, probabilities_est, rules, outcome, eps=0.005):
        """
        estimation set for non-separate case
        """
        est_result = {}
        if 'left_result' in result:
            est_result['feature'] = result['feature']
            est_result['threshold'] = result['threshold']
            est_result['left_result'] = self._split_exposure_validate_eht(node_id*2, df_est, result['left_result'], probabilities_est, 
                                                       rules+[(result['feature'], 0, result['threshold'])], outcome, eps)
            est_result['right_result'] = self._split_exposure_validate_eht(node_id*2+1, df_est, result['right_result'], probabilities_est, 
                                                         rules+[(result['feature'], 1, result['threshold'])], outcome, eps)
        
        if rules:
            # if this is not the root
            idxs = np.product([df_est[key] <= th for key, sign, th in rules if sign == 0] + \
                   [df_est[key] > th for key, sign, th in rules if sign == 1],
                   axis=0) > 0
            dff = df_est[idxs]
        else:
            idxs = np.ones(len(df_est)).astype(bool)
            dff = df_est
            
        propensities_1 = np.mean(np.product([probabilities_est[key][idxs] <= th for key, sign, th in rules if sign == 0] + \
               [probabilities_est[key][idxs] > th for key, sign, th in rules if sign == 1]+\
                [probabilities_est[self.treatment][idxs] == 1],
               axis=0), axis=1)

        propensities_0 = np.mean(np.product([probabilities_est[key][idxs] <= th for key, sign, th in rules if sign == 0] + \
               [probabilities_est[key][idxs] > th for key, sign, th in rules if sign == 1]+\
               [probabilities_est[self.treatment][idxs] == 0],
               axis=0), axis=1)
            
        idxs_filter = np.logical_and(propensities_1 > 0, propensities_0 > 0)
        dff = dff[idxs_filter]
        propensities_1 = propensities_1[idxs_filter]
        propensities_0 = propensities_0[idxs_filter]
                
        mod = sm.WLS(dff[outcome], sm.add_constant(dff[self.treatment]), 
                     weights=1.0 / propensities_1 * dff[self.treatment] + 1.0 / propensities_0 * (1-dff[self.treatment]))        
        res = mod.fit()
        mse = np.sum((res.resid ** 2) * (1.0 / propensities_1 * dff[self.treatment] + 1.0 / propensities_0 * (1-dff[self.treatment]))) 
        
        average_hajek = res.params[1]
        average_hajek_se = res.bse[1] # dff[outcome].std() / np.sqrt(len(dff)-1)
        
        est_result['hajek'] = average_hajek
        est_result['hajek_se'] = average_hajek_se
        est_result['mse'] = mse
        est_result['N'] = len(dff)
        return est_result
    
    def _split_exposure_validate(self, node_id, df_est, result, 
                                 probabilities_est, rules, outcome, eps=0.005):

        est_result = {}
        if 'left_result' in result:
            est_result['feature'] = result['feature']
            est_result['threshold'] = result['threshold']
            est_result['left_result'] = self._split_exposure_validate(node_id*2, df_est, result['left_result'], probabilities_est, 
                                                       rules+[(result['feature'], 0, result['threshold'])], outcome, eps)
            est_result['right_result'] = self._split_exposure_validate(node_id*2+1, df_est, result['right_result'], probabilities_est, 
                                                         rules+[(result['feature'], 1, result['threshold'])], outcome, eps)
        
        if rules:
            idxs = np.product([df_est[key] <= th for key, sign, th in rules if sign == 0] + \
                   [df_est[key] > th for key, sign, th in rules if sign == 1], axis=0) > 0
            dff = df_est[idxs]
            propensities = np.mean(np.product([probabilities_est[key][idxs] <= th for key, sign, th in rules if sign == 0] + \
                   [probabilities_est[key][idxs] > th for key, sign, th in rules if sign == 1],
                   axis=0), axis=1)
            idxs_filter = propensities > eps
            dff = dff[idxs_filter]
            propensities = propensities[idxs_filter]
        else:
            dff = df_est
            propensities = np.ones(len(dff))
        
        mod = sm.OLS(dff[outcome], np.ones(len(dff)))        
        res = mod.fit()
        mse = np.sum((res.resid ** 2) * 1.0 / propensities)
        average_hajek = res.params[0]
        
        if node_id == 1:
            average_hajek_se = dff[outcome].std() / np.sqrt(len(dff)-1)
        else:
            average_hajek_se = self._hajek_se(dff, propensities, outcome)
        
        est_result['hajek'] = average_hajek
        est_result['hajek_se'] = average_hajek_se
        est_result['mse'] = mse
        est_result['N'] = len(dff)
        return est_result
    
    def _split_exposure_hajek_eht(self, node_id, df, probabilities, feature_set, max_attempt, eps, delta, outcome, rules, N, current_mse, criteria):
        """
        the actual splitting implementation for non-separate tree; 
        recursion
        """
        b_feature = ''
        b_threshold = 0
        b_left = None
        b_right = None
        b_average_left_hajek = 0
        b_average_right_hajek = 0
        b_mse = 10000000000.0

        ranges = {}
        
        for feature in feature_set:
            gc.collect()
            # find the more compact valid region
            upper = 1.
            lower = 0.
            for rule in rules:
                if rule[0] == feature:
                    if rule[1] == 0:
                        lower = np.maximum(rule[2], lower)
                    else:
                        upper = np.minimum(rule[2], upper)
            
            if lower > upper:
                continue
                
            for k in range(max_attempt):
                threshold = np.random.uniform(lower, upper)
                
                cz_l_1 = self._contain_zero(probabilities, rules+[(feature, 0, threshold)], eps, delta, treated=1)
                cz_r_1 = self._contain_zero(probabilities, rules+[(feature, 1, threshold)], eps, delta, treated=1)
                cz_l_0 = self._contain_zero(probabilities, rules+[(feature, 0, threshold)], eps, delta, treated=0)
                cz_r_0 = self._contain_zero(probabilities, rules+[(feature, 1, threshold)], eps, delta, treated=0)
                
                if np.mean(cz_l_1) > delta or np.mean(cz_r_1) > delta or np.mean(cz_r_0) > delta or np.mean(cz_r_0) > delta:
                    continue

                idxs_left = np.product([df[key] <= th for key, sign, th in rules if sign == 0] + \
                       [df[key] > th for key, sign, th in rules if sign == 1] + \
                        [df[feature] <= threshold],
                       axis=0) > 0

                idxs_right = np.product([df[key] <= th for key, sign, th in rules if sign == 0] + \
                       [df[key] > th for key, sign, th in rules if sign == 1] + \
                        [df[feature] > threshold],
                       axis=0) > 0

                left = df[idxs_left]
                right = df[idxs_right]
                
                # propensity score for left partition + ego treated
                propensities_left_1 = np.mean(np.product([probabilities[key][idxs_left] <= th for key, sign, th in rules if sign == 0] + \
                            [probabilities[key][idxs_left] > th for key, sign, th in rules if sign == 1] + \
                            [probabilities[feature][idxs_left] <= threshold] + \
                            [probabilities[self.treatment][idxs_left] == 1],
                           axis=0), axis=1)
                
                # propensity score for left partition + ego non treated
                propensities_left_0 = np.mean(np.product([probabilities[key][idxs_left] <= th for key, sign, th in rules if sign == 0] + \
                            [probabilities[key][idxs_left] > th for key, sign, th in rules if sign == 1] + \
                            [probabilities[feature][idxs_left] <= threshold] + \
                            [probabilities[self.treatment][idxs_left] == 0],
                           axis=0), axis=1)
                
                propensities_right_1 = np.mean(np.product([probabilities[key][idxs_right] <= th for key, sign, th in rules if sign == 0] + \
                            [probabilities[key][idxs_right] > th for key, sign, th in rules if sign == 1] + \
                            [probabilities[feature][idxs_right] > threshold] + \
                            [probabilities[self.treatment][idxs_right] == 1],
                           axis=0), axis=1)

                propensities_right_0 = np.mean(np.product([probabilities[key][idxs_right] <= th for key, sign, th in rules if sign == 0] + \
                            [probabilities[key][idxs_right] > th for key, sign, th in rules if sign == 1] + \
                            [probabilities[feature][idxs_right] > threshold] + \
                            [probabilities[self.treatment][idxs_right] == 0],
                           axis=0), axis=1)
                
                # filter those whose propensities scores are very small (This may lead to lose observations)
                idxs_left_filter = np.logical_and(propensities_left_1 > eps, propensities_left_0 > eps)
                left = left[idxs_left_filter]
                propensities_left_1 = propensities_left_1[idxs_left_filter]
                propensities_left_0 = propensities_left_0[idxs_left_filter]
                
                # filter those whose propensities scores are very small (This may lead to lose observations)
                idxs_right_filter = np.logical_and(propensities_right_1 > eps, propensities_right_0 > eps)
                right = right[idxs_right_filter]
                propensities_right_1 = propensities_right_1[idxs_right_filter]                
                propensities_right_0 = propensities_right_0[idxs_right_filter]                
                
                if np.mean(left[self.treatment]) == 0 or np.mean(left[self.treatment]) == 1 or \
                    np.mean(right[self.treatment]) == 0 or np.mean(right[self.treatment]) == 1:
                    continue
                
                if len(left) == 0 or len(right) == 0:
                    continue
                
                # The covariate implementation does not work as expected; should always be None

                mod_left = sm.WLS(left[outcome], sm.add_constant(left[[self.treatment]]), \
                    weights=1.0 / propensities_left_1 * left[self.treatment] + 1.0 / propensities_left_0 * (1-left[self.treatment]))
                res_left = mod_left.fit()

                mod_right = sm.WLS(right[outcome], sm.add_constant(right[self.treatment]), \
                    weights=1.0 / propensities_right_1 * right[self.treatment] + 1.0 / propensities_right_0 * (1-right[self.treatment]))
                res_right = mod_right.fit()
                
                average_left_hajek = res_left.params[1] 
                average_right_hajek = res_right.params[1]

                average_left_hajek_se = res_left.bse[1]  
                average_right_hajek_se = res_right.bse[1]
                # need further improvement
                
                mse_left = np.sum((1.0 / propensities_left_1 * left[self.treatment] + 1.0 / propensities_left_0 * (1-left[self.treatment])) * 
                                  ((res_left.resid) ** 2))
                mse_right = np.sum((1.0 / propensities_right_1 * right[self.treatment] + 1.0 / propensities_right_0 * (1-right[self.treatment])) * 
                                   ((res_right.resid) ** 2))
                mse = mse_left * 1.0 * len(left)/(len(left)+len(right)) + mse_right * 1.0 * len(right)/(len(left)+len(right))

                if mse < b_mse:
                    flag = True
                    assert len(criteria) > 0
                    if 'non_trivial_reduction' in criteria:
                        if not (mse < current_mse - criteria['non_trivial_reduction']):
                            flag = False
                    if 'reasonable_propensity' in criteria:
                        if not (np.abs(np.sum(1.0 / propensities_left_1 * left[self.treatment])/len(df) - 1.0) <= criteria['reasonable_propensity'] \
                                and \
                                np.abs(np.sum(1.0 / propensities_right_1 * right[self.treatment])/len(df) - 1.0) <= criteria['reasonable_propensity'] \
                                and \
                                np.abs(np.sum(1.0 / propensities_left_0 * (1 - left[self.treatment]))/len(df) - 1.0) <= criteria['reasonable_propensity'] \
                                and \
                                np.abs(np.sum(1.0 / propensities_right_0 * (1 - right[self.treatment]))/len(df) - 1.0) <= criteria['reasonable_propensity']
                               ):
                            flag = False
                    if 'separate_reduction' in criteria:
                        if not (mse_left < current_mse and mse_right < current_mse):
                            flag = False
                    if 'min_leaf_size' in criteria:
                        if not (len(left) >= criteria['min_leaf_size'] and len(right) >= criteria['min_leaf_size']):
                            flag = False
                    if flag:
                        b_feature = feature
                        b_mse = mse
                        b_mse_left = mse_left
                        b_mse_right = mse_right
                        b_threshold = threshold
                        b_average_left_hajek = average_left_hajek
                        b_average_right_hajek = average_right_hajek
                        b_average_left_hajek_se = average_left_hajek_se
                        b_average_right_hajek_se = average_right_hajek_se
                        b_left = left
                        b_right = right
                        b_left_rules = rules + [(feature, 0, threshold)]
                        b_right_rules = rules + [(feature, 1, threshold)]
        
        result = {}
        if b_feature != '':
            # if find a valid partition
            result_left = self._split_exposure_hajek_eht(node_id*2, df, probabilities, feature_set, max_attempt, eps, delta, outcome, b_left_rules, N, b_mse_left, criteria)       
            result_right = self._split_exposure_hajek_eht(node_id*2+1, df, probabilities, feature_set, max_attempt, eps, delta, outcome, b_right_rules, N, b_mse_right, criteria)

            result['mse'] = result_left['mse'] * 1.0 * len(b_left)/(len(b_left)+len(b_right)) + \
                        result_right['mse'] * 1.0 * len(b_right)/(len(b_left)+len(b_right))
            result['feature'] = b_feature
            result['threshold'] = b_threshold
            result_left['hajek'] = b_average_left_hajek
            result_right['hajek'] = b_average_right_hajek
            result_left['hajek_se'] = b_average_left_hajek_se
            result_right['hajek_se'] = b_average_right_hajek_se
            result_left['N'] = len(b_left)
            result_right['N'] = len(b_right)
            result['left_result'] = result_left
            result['right_result'] = result_right
            return result
        else:
            result['mse'] = current_mse
            return result
        
    def estimate_exposure_hajek(self, train_result_separate, indirect_space, outcome, eps=0.005, separate=True):
        """
        train_result_separate: result from training
        indirect_space: feature space (consistent with training input)
        outcome: (consistent with training input)
        eps: (consistent with training input)
        df_est=None: leave it 
        probabilities=None: leave it 
        separate=True: separate trees.
        """
        if separate:
            df_est = self.df_est
            probabilities = self.probabilities

            probabilities_est = {}
            for key in [self.treatment]+indirect_space:
                probabilities_est[key] = self.probabilities[key][self.idx_est]

            est_result_separate = {} 
            est_result_separate = self._split_exposure_validate(1, df_est, train_result_separate, probabilities_est, [], outcome, eps)
            self.est_result_separate = est_result_separate
            return est_result_separate
        else:
            # if find a valid partition for T == 1 or 0 separately
            df_est = self.df_est
            probabilities_est = {}
            for key in indirect_space+[self.treatment]:
                    probabilities_est[key] = self.probabilities[key][self.idx_est.astype(bool)]
            est_result_separate_eht = {} 
            est_result_separate_eht = self._split_exposure_validate_eht(1, df_est, train_result_separate, probabilities_est, [], outcome, eps)
            self.est_result_separate_eht = est_result_separate_eht
            return est_result_separate_eht
        
    def split_exposure_hajek(self, separate, outcome, feature_set, max_attempt=30, eps=0.0, delta=0.0, 
                             df_train=None, probabilities=None, criteria={'non_trivial_reduction': 0}):
        """
        The API for spitting
        separate: True=separate trees
        outcome: outcome variable
        feature_set: a list of features used to partition (may include ``assignment'')
        min_variance_reduct: minimum variance reduction in each partition, only partition if reduction is significantly large
        max_attempt: sample threshold -- a larger value tend to over fit more
        eps: avoid non-zero or zero-trivial probability
        delta: avoid non-zero or zero-trivial probability
        df_train: leave it as None
        probabilities: leave it as None
        """
        if separate == True:
            df_train = self.df_train  # training set
            probabilities = self.probabilities  # probability tensor
            
            probabilities_train = {}
            
            for key in [self.treatment]+feature_set:
                probabilities_train[key] = probabilities[key][self.idx_tr]

            mod = sm.WLS(df_train[outcome], np.ones(len(df_train)))
            res = mod.fit()
            total_sse = np.sum(res.resid ** 2)  # total sse

            train_result = {}
            train_result = self._split_exposure_hajek(1, df_train, probabilities_train, feature_set, max_attempt, 
                                                      eps, delta, outcome, [], 
                                                      len(df_train), total_sse, criteria)
            train_result['N'] = len(df_train)
            train_result['hajek'] = df_train[outcome].mean()
            train_result['hajek_se'] = df_train[outcome].std() / np.sqrt(len(df_train[outcome])-1)
            self.result_separate = train_result
            return train_result
        else:
            df_train = self.df_train
            probabilities = self.probabilities
            probabilities_train = {}
            for key in [self.treatment]+feature_set:
                probabilities_train[key] = probabilities[key][self.idx_tr]
            mod = sm.WLS(df_train[outcome], sm.add_constant(df_train[self.treatment]))

            res = mod.fit()
            total_sse = np.sum(res.resid ** 2) * 2

            train_result_eht = {}
            train_result_eht = self._split_exposure_hajek_eht(1, df_train, probabilities_train, feature_set, max_attempt, 
                                                              eps, delta, outcome, [], len(df_train), total_sse, criteria)
            train_result_eht['N'] = len(df_train)
            train_result_eht['hajek'] = res.params[1]
            train_result_eht['hajek_se'] = res.bse[1]
            return train_result_eht
        

