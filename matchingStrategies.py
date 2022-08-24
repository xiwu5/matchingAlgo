import pandas as pd
import itertools
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype
import logging
from ast import literal_eval

logger = logging.getLogger('MatchingStrategies')
logger.setLevel(logging.DEBUG)

class MatchingStrategies:
    def __init__(self):
        self.hardConstraintsAlgoMap = {
            "SIMILAR": self.similarFilter,
            "MAX": self.sumFilter,
            "RANGE": self.rangeFilter,
            "WITHIN": self.withinFilter
        }
        self.ratingAlgoMap = {
            "SIMILAR": self.similarRator,
            "EXCLUDE": self.excludeRator,
            "WITHIN": self.withinRator,
            "MAX": self.sumRator,
            "EXTERNAL_RATOR": self.externalRator
        }
    def minMaxNormalize(self, df, invert = False):
        if ((df.max()-df.min()) == 0):
            return np.zeros_like(df)
        norm = (df-df.min())/(df.max()-df.min())
        if not invert:
            return norm
        return 1-norm
    
    def withinFilter(self, userMatching, userMatched, optional):
        logger.debug('Within filter called')
        if (len(optional["extraData"]) == 2):
            userMatchingExpected, userMatchedExpected = optional["extraData"]
        indexRet = pd.Index([]).astype('int64')
        if (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            try:
                userMatchingSet = userMatching.apply(lambda x: set(x.split(', ')))
                userMatchedSet = userMatched.apply(lambda x: set(x.split(', ')))
            except:
                userMatchingSet = userMatching.apply(lambda x: set([x]))
                userMatchedSet = userMatched.apply(lambda x: set([x]))
            try:
                userMatchingExpectedSet = userMatchingExpected.apply(lambda x: set(x.split(', ')))
                userMatchedExpectedSet = userMatchedExpected.apply(lambda x: set(x.split(', ')))
            except:
                userMatchingExpectedSet = userMatchingExpected.apply(lambda x: set([x]))
                userMatchedExpectedSet = userMatchedExpected.apply(lambda x: set([x]))
            def shouldFilter(x):
                if (('pass' in optional) and len(x[1].intersection(set(optional['pass'])))>0):
                    return False
                return len(x[0].intersection(x[1]))<=0
            if (optional["direction"] == "both" or optional["direction"] == "userMatching<userMatched"):
                filtered = pd.concat([userMatchedSet, userMatchingExpectedSet], axis=1).apply(shouldFilter, axis=1)
                indexRet = indexRet.union(userMatching[filtered].index)
            if (optional["direction"] == "both" or optional["direction"] == "userMatching>userMatched"):
                filtered = pd.concat([userMatchingSet, userMatchedExpectedSet], axis=1).apply(shouldFilter, axis=1)
                indexRet = indexRet.union(userMatching[filtered].index)
            return indexRet
        else:
            logger.error("Un-recognized data type or optional args are incorrect")
            return indexRet

    
    def similarFilter(self, userMatching, userMatched, optional):
        if (is_numeric_dtype(userMatching) and is_numeric_dtype(userMatched)):
            diff = (userMatching-userMatched).abs()
            return diff[diff > optional["threshold"]].index
        elif (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            try:
                matchingSet = userMatching.apply(lambda x: set(x.split(', ')))
                matchedSet = userMatched.apply(lambda x: set(x.split(', ')))
            except:
                matchingSet = userMatching.apply(lambda x: set([x]))
                matchedSet = userMatched.apply(lambda x: set([x]))
            def findIntersectLen(x):
                if (('pass' in optional) and (len(x[0].intersection(set(optional['pass'])))>0 or len(x[1].intersection(set(optional['pass'])))>0)):
                    return optional["threshold"]
                return len(x[0].intersection(x[1]))
            similar = pd.concat([matchingSet, matchedSet], axis=1).apply(findIntersectLen, axis=1)
            return similar[similar < optional["threshold"]].index

    def sumFilter(self, userMatching, userMatched, optional):
        if (is_numeric_dtype(userMatching) and is_numeric_dtype(userMatched)):
            userSum = (userMatching+userMatched).abs()
            return userSum[userSum > optional["threshold"]].index
        elif (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            matchingSet = userMatching.apply(lambda x: set(x.split(', ')))
            matchedSet = userMatched.apply(lambda x: set(x.split(', ')))
            def findUnionLen(x):
                return len(x[0].union(x[1]))
            comb = pd.concat([matchingSet, matchedSet], axis=1).apply(findUnionLen, axis=1)
            return comb[comb < optional["threshold"]].index
    def rangeFilter(self, userMatching, userMatched, optional):
        if (len(optional["extraData"]) != 4):
            logger.error("Not enough data for range operation.")
            return pd.Index([]).astype('int64')
        userMatchingMin, userMatchingMax, userMatchedMin, userMatchedMax = optional["extraData"]
        if (not all([is_numeric_dtype(x) for x in [userMatching, userMatchingMin, userMatchingMax, userMatched, userMatchedMin, userMatchedMax]])):
            logger.error("Non-numerical columns detected for rangeFilter")
            return pd.Index([]).astype('int64')
        indexRet = pd.Index([]).astype('int64')
        if (optional["direction"] == "both" or optional["direction"] == "userMatching<userMatched"):
            indexRet = indexRet.union(userMatched[(userMatched <= userMatchingMin) | (userMatched >= userMatchingMax)].index)
        if (optional["direction"] == "both" or optional["direction"] == "userMatching>userMatched"):
            indexRet = indexRet.union(userMatching[(userMatching <= userMatchedMin) | (userMatching >= userMatchedMax)].index)
        return indexRet

    
    def similarRator(self, userMatching, userMatched, optional):
        logger.debug("similarRator invoked")
        logger.debug(userMatching)
        logger.debug(userMatched)
        if (is_bool_dtype(userMatching) and is_bool_dtype(userMatched)):
            diff = (userMatching.astype(int)-userMatched.astype(int)).abs()
            return self.minMaxNormalize(diff, True)
        elif (is_numeric_dtype(userMatching) and is_numeric_dtype(userMatched)):
            diff = (userMatching-userMatched).abs()
            return self.minMaxNormalize(diff, True)
        elif (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            try:
                matchingSet = userMatching.apply(lambda x: set(x.split(', ')))
                matchedSet = userMatched.apply(lambda x: set(x.split(', ')))
            except:
                matchingSet = userMatching.apply(lambda x: set([x]))
                matchedSet = userMatched.apply(lambda x: set([x]))
            def findIntersectLen(x):
                if (('pass' in optional) and len(x[1].intersection(set(optional['pass'])))>0):
                    return 1
                return len(x[0].intersection(x[1]))
            similar = pd.concat([matchingSet, matchedSet], axis=1).apply(findIntersectLen, axis=1)
            return self.minMaxNormalize(similar, False)

    def excludeRator(self, userMatching, userMatched, optional):
        logger.debug('Exclude rator called')
        if (len(optional["extraData"]) == 2):
            userMatchingNotExpected, userMatchedNotExpected = optional["extraData"]
        if (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            try:
                userMatchingSet = userMatching.apply(lambda x: set(x.split(', ')))
                userMatchedSet = userMatched.apply(lambda x: set(x.split(', ')))
            except:
                userMatchingSet = userMatching.apply(lambda x: set([x]))
                userMatchedSet = userMatched.apply(lambda x: set([x]))
            try:
                userMatchingNotExpectedSet = userMatchingNotExpected.apply(lambda x: set(x.split(', ')))
                userMatchedNotExpectedSet = userMatchedNotExpected.apply(lambda x: set(x.split(', ')))
            except:
                userMatchingNotExpectedSet = userMatchingNotExpected.apply(lambda x: set([x]))
                userMatchedNotExpectedSet = userMatchedNotExpected.apply(lambda x: set([x]))
            def findIntersectLen(x):
                if (('pass' in optional) and len(x[1].intersection(set(optional['pass'])))>0):
                    return 0
                return len(x[0].intersection(x[1]))
            similar = np.zeros(userMatchedSet.shape)
            if (optional["direction"] == "both" or optional["direction"] == "userMatching<userMatched"):
                similar = np.add(similar, pd.concat([userMatchedSet, userMatchingNotExpectedSet], axis=1).apply(findIntersectLen, axis=1))
            if (optional["direction"] == "both" or optional["direction"] == "userMatching>userMatched"):
                similar = np.add(similar, pd.concat([userMatchingSet, userMatchedNotExpectedSet], axis=1).apply(findIntersectLen, axis=1))
            return self.minMaxNormalize(similar, True)
    def withinRator(self, userMatching, userMatched, optional):
        logger.debug('WITHIN rator called')
        if (len(optional["extraData"]) == 2):
            userMatchingNotExpected, userMatchedNotExpected = optional["extraData"]
        if (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            try:
                userMatchingSet = userMatching.apply(lambda x: set(x.split(', ')))
                userMatchedSet = userMatched.apply(lambda x: set(x.split(', ')))
            except:
                userMatchingSet = userMatching.apply(lambda x: set([x]))
                userMatchedSet = userMatched.apply(lambda x: set([x]))
            try:
                userMatchingExpectedSet = userMatchingNotExpected.apply(lambda x: set(x.split(', ')))
                userMatchedExpectedSet = userMatchedNotExpected.apply(lambda x: set(x.split(', ')))
            except:
                userMatchingExpectedSet = userMatchingNotExpected.apply(lambda x: set([x]))
                userMatchedExpectedSet = userMatchedNotExpected.apply(lambda x: set([x]))
            def findIntersectLen(x):
                if (('pass' in optional) and len(x[1].intersection(set(optional['pass'])))>0 ):
                    if ('one_is_enough' in optional and optional['one_is_enough']):
                        return 1
                    return np.nan
                if ('one_is_enough' in optional and optional['one_is_enough']):
                    return 1 if len(x[0].intersection(x[1]))> 0 else 0
                return len(x[0].intersection(x[1]))
            similar = np.zeros(userMatchedSet.shape)
            if (optional["direction"] == "both" or optional["direction"] == "userMatching<userMatched"):
                similar = np.add(similar, pd.concat([userMatchedSet, userMatchingExpectedSet], axis=1).apply(findIntersectLen, axis=1))
            if (optional["direction"] == "both" or optional["direction"] == "userMatching>userMatched"):
                similar = np.add(similar, pd.concat([userMatchingSet, userMatchedExpectedSet], axis=1).apply(findIntersectLen, axis=1))
            similar.fillna(value=similar.mean(), inplace=True)
            return self.minMaxNormalize(similar, False)

    def sumRator(self, userMatching, userMatched, optional):
        if (is_numeric_dtype(userMatching) and is_numeric_dtype(userMatched)):
            userSum = (userMatching+userMatched).abs()
            return self.minMaxNormalize(userSum, False)
        elif (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            matchingSet = userMatching.apply(lambda x: set(x.split(', ')))
            matchedSet = userMatched.apply(lambda x: set(x.split(', ')))
            def findUnionLen(x):
                return len(x[0].union(x[1]))
            comb = pd.concat([matchingSet, matchedSet], axis=1).apply(findUnionLen, axis=1)
            return self.minMaxNormalize(comb, False)

    def externalRator(self, userMatching, userMatched, optional):
        externalChart = pd.read_csv(optional["fileName"], index_col=0, header=0)
        if (is_string_dtype(userMatching) and is_string_dtype(userMatched)):
            def findScore(x):
                if (('pass' in optional) and ((not isinstance(x[0], str)) or (not isinstance(x[1], str)) or x[0] in optional['pass'] or x[1] in optional['pass']) ):
                    return np.nan
                print(x[0], x[1])
                return externalChart.loc[x[0], x[1]]
            score = np.zeros(userMatching.shape)
            if (optional["direction"] == "both" or optional["direction"] == "userMatching<userMatched"):
                score = np.add(score, pd.concat([userMatching, userMatched], axis=1).apply(findScore, axis=1))
            if (optional["direction"] == "both" or optional["direction"] == "userMatching>userMatched"):
                score = np.add(score, pd.concat([userMatched, userMatching], axis=1).apply(findScore, axis=1))
            score.fillna(value=score.mean(), inplace=True)
            return self.minMaxNormalize(score, False)
