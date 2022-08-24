import pandas as pd
import itertools
import numpy as np
import matchingStrategies as ms
import json
import logging
import jinja2
import collections

logger = logging.getLogger('DataProcessor')
logger.setLevel(logging.DEBUG)
class DataProcessor:
    def __init__(self):
        self.df = []
        self.pairedDf = []
        self.filteredDf = []
        self.finalPairs = []
        self.scoreBoard = []
        self.ratedWeight = []
        self.strategies = {}
        self.dropped = {}
        self.strategiesEngine = ms.MatchingStrategies()

    def loadDataFromFile(self, fileName, strategyName):
        self.df = pd.read_csv(fileName, dtype = str)
        with open(strategyName) as strategiesFile: 
            self.strategies = json.load(strategiesFile) 
        self.mapToPairs()

    def mapToPairs(self):
        a,b = map(list, zip(*itertools.combinations(self.df.index, 2)))
        merged = pd.concat([self.df.loc[a].reset_index(drop=True), self.df.loc[b].reset_index(drop=True)], axis=1)
        merged.columns = list(map('.'.join, itertools.product(["matchingUser", "matchedUser"], ['UserName'] + list(self.df.columns)[1:])))
        self.pairedDf = merged.infer_objects()
        logger.info("Finished mapToPairs, remaining entries")
        logger.info(self.pairedDf)
        logger.info(self.pairedDf.shape)

    def removeImpossiblePairs(self):
        self.filteredDf = self.pairedDf.copy()
        for ruleColumn in self.strategies["hardConstraints"]:
            logger.info("Processing rule named {}".format(ruleColumn))
            if (('matchingUser.{}'.format(ruleColumn) not in set(self.filteredDf.columns)) or ('matchedUser.{}'.format(ruleColumn) not in set(self.filteredDf.columns))):
                logger.info('{} not in existing columns, skipping this rule'.format(ruleColumn))
                continue
            ruleType = self.strategies["hardConstraints"][ruleColumn]["ruleType"]
            if (ruleType in self.strategiesEngine.hardConstraintsAlgoMap):
                options = self.strategies["hardConstraints"][ruleColumn].copy()
                if ("postfixes" in options):
                    extraColumnNames = list(map('.'.join, itertools.product(["matchingUser", "matchedUser"], ['{}{}'.format(ruleColumn, post) for post in options["postfixes"]])))
                    options["extraData"] = [self.filteredDf[x] for x in extraColumnNames]
                needDrop = self.strategiesEngine.hardConstraintsAlgoMap[ruleType](self.filteredDf['matchingUser.{}'.format(ruleColumn)], self.filteredDf['matchedUser.{}'.format(ruleColumn)], optional = options)
                self.dropped[ruleColumn] = self.filteredDf.iloc[needDrop, :].copy()
                self.filteredDf = self.filteredDf.drop(needDrop, axis=0).reset_index(drop=True)
            else:
                logger.info('Method for {} : {} not yet implemented'.format(ruleColumn, ruleType))
        logger.info("Finished removeImpossiblePairs, remaining entries")
        logger.info(self.filteredDf.shape)

    # this function computes ratings for created pairs based on 
    def computeRatingsForPairs(self):
        logger.debug("computeRatingsForPairs for dataset")
        filteredDf = self.filteredDf.copy()
        logger.debug(filteredDf.columns)
        self.ratedWeight = []
        self.scoreBoard = filteredDf[['matchingUser.UserName', 'matchedUser.UserName']]
        if (filteredDf.shape[0] == 0):
            return
        for ruleColumn in self.strategies["ratedConstraints"]:
            logger.info("Processing rule named {}".format(ruleColumn))
            if (('matchingUser.{}'.format(ruleColumn) not in set(filteredDf.columns)) or ('matchedUser.{}'.format(ruleColumn) not in set(filteredDf.columns))):
                logger.info('{} not in existing columns, skipping this rule'.format(ruleColumn))
                continue
            ruleType = self.strategies["ratedConstraints"][ruleColumn]["ruleType"]
            if (ruleType in self.strategiesEngine.ratingAlgoMap):
                options = self.strategies["ratedConstraints"][ruleColumn].copy()
                if ("postfixes" in options):
                    extraColumnNames = list(map('.'.join, itertools.product(["matchingUser", "matchedUser"], ['{}{}'.format(ruleColumn, post) for post in options["postfixes"]])))
                    options["extraData"] = [filteredDf[x] for x in extraColumnNames]
                score = self.strategiesEngine.ratingAlgoMap[ruleType](filteredDf['matchingUser.{}'.format(ruleColumn)], filteredDf['matchedUser.{}'.format(ruleColumn)], optional = options)
                print(score)
                scoreColumnName = '{}Score'.format(ruleColumn)
                self.scoreBoard = self.scoreBoard.assign(**{scoreColumnName: score.values})
                self.ratedWeight.append(self.strategies["ratedConstraints"][ruleColumn]["weight"])
            else:
                logger.info('Method for {} : {} not yet implemented'.format(ruleColumn, ruleType))
        self.scoreBoard["totalScore"] = np.matmul(np.array(self.scoreBoard.iloc[:, 2:]), np.array(self.ratedWeight))
        self.scoreBoard = self.scoreBoard.sort_values(by=['totalScore'], ascending = False)
    def getPairs(self):
        pairNumSoFar = 0
        existingUser = set()
        pairIndex = []
        logger.debug('Final score board:')
        logger.debug(self.scoreBoard)
        logger.debug(self.scoreBoard.shape)
        for i in range(self.scoreBoard.shape[0]):
            matchingUser, matchedUser = self.scoreBoard[['matchingUser.UserName', 'matchedUser.UserName']].iloc[i]
            if ( (matchingUser not in existingUser) and (matchedUser not in existingUser)):
                pairNumSoFar += 1
                pairIndex.append(i)
                existingUser.update([matchingUser, matchedUser])
            if (pairNumSoFar == self.strategies["totalPairs"]):
                logger.info("Required {} pairs found, end searching".format(self.strategies["totalPairs"]))
                break
        self.finalPairs = self.scoreBoard.iloc[pairIndex]
        self.finalPairs = self.finalPairs.merge(self.df, how='inner', left_on="matchingUser.UserName", right_on="UserName", suffixes=('', 'MatchingUser'))
        self.finalPairs = self.finalPairs.merge(self.df, how='inner', left_on="matchedUser.UserName", right_on="UserName", suffixes=('', 'MatchedUser'))
        self.finalPairs = self.finalPairs[list(self.finalPairs.columns[:3+len(self.ratedWeight)]) + list(sorted(self.finalPairs.columns[3+len(self.ratedWeight): ]))]
        self.scoreBoard = self.scoreBoard.merge(self.df, how='inner', left_on="matchingUser.UserName", right_on="UserName", suffixes=('', 'MatchingUser'))
        self.scoreBoard = self.scoreBoard.merge(self.df, how='inner', left_on="matchedUser.UserName", right_on="UserName", suffixes=('', 'MatchedUser'))
        self.scoreBoard = self.scoreBoard[list(self.scoreBoard.columns[:3+len(self.ratedWeight)]) + list(sorted(self.scoreBoard.columns[3+len(self.ratedWeight): ]))]

    def generateReports(self):
        # output final pairs
        self.finalPairs.to_csv("../output/finalPairs.csv")
        self.scoreBoard.to_csv("../output/scoreBoard.csv")
        for key in self.dropped:
            self.dropped[key].to_csv("../output/droppedBy{}.csv".format(key))


        # create a more detailed report
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='../configs'))
        template = env.get_template('reportTemplate.html')

        # get # of dropped pairs by feature
        self.finalDroppedTable = pd.DataFrame.from_dict(self.dropped, orient='index')
        droppedCounter = {}
        total = self.pairedDf.shape[0]
        droppedCounter = {"totalPairsPossible": total}
        for element in self.dropped:
            droppedCounter[element] = self.dropped[element].shape[0]
            total -= droppedCounter[element]
        droppedCounter["remaining"] = total
        droppedCount = pd.DataFrame.from_dict(droppedCounter, orient='index')
        droppedCount.columns = ['#pairsDropped']

        # get most matched users
        allUsersMatched = list(self.scoreBoard['matchingUser.UserName']) + list(self.scoreBoard['matchedUser.UserName'])
        dict(collections.Counter(allUsersMatched))
        userMatchedCountTable = pd.DataFrame.from_dict(dict(collections.Counter(allUsersMatched)), orient='index').reset_index()
        try:
            userMatchedCountTable.columns = ["userName", "matchCount"]
            userTableWithMatchCount = self.df.merge(userMatchedCountTable, how='left', left_on="Unnamed: 0", right_on="userName")
            userTableWithMatchCountSorted = userTableWithMatchCount.sort_values(by=['matchCount'], ascending = False)
        except:
            userTableWithMatchCountSorted = userMatchedCountTable
        html = template.render(
            table=self.finalPairs.to_html(),
            droppedCountTable=droppedCount.to_html(),
            userByMatchCount = userTableWithMatchCountSorted.iloc[:, :].to_html()
        )
        with open('../output/matchingReport.html', 'w+') as f:
            f.write(html)

        


            
