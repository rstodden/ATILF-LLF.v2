from enum import Enum

from param import XPParams


class TransitionType(Enum):
    SHIFT = 0
    MERGE = 1
    COMPLETE = 2
    MWT_COMPLETE = 3
    REDUCE = 4
    WHITE_MERGE = 5
    MERGE_AS_VPC_FULL = 6
    MERGE_AS_VPC_SEMI = 7
    MERGE_AS_IReflV = 8
    MERGE_AS_OTH = 9
    MERGE_AS_ID = 10
    MERGE_AS_LVC_FULL = 11
    MERGE_AS_LVC_CAUSE = 12
    MERGE_AS_MVC = 14
    MERGE_AS_IAV = 15
    MERGE_AS_LS_ICV = 16

    @staticmethod
    def getAllClasses():
        if not XPParams.includeEmbedding:
            return [0, 1, 2, 3]
        allClasses = []
        for tType in EmbedTransType:
            allClasses.append(tType.value)
        for tType in MWTTransitionType:
            allClasses.append(tType.value)
        return allClasses

    @staticmethod
    def getType(idx):
        """
        :param idx: integer of transition
        :return: if transition in TransitionsType or MWTTransitionType then return object
        else return None
        """
        for tType in TransitionType:
            if tType.value == idx:
                return tType
        for tType in MWTTransitionType:
            if tType.value == idx:
                return tType
        return None

    @staticmethod
    def inZeroCostTrans(transTypeValue, zeroCostTrans):

        for elem in zeroCostTrans:
            if transTypeValue == elem.value:
                return True
        return False

    @staticmethod
    def sort(transTypes):
        result = []

        if not XPParams.includeEmbedding:
            for elem in transTypes:
                if elem == TransitionType.MWT_COMPLETE:
                    result.append(elem)
            for elem in transTypes:
                if elem == TransitionType.MERGE:
                    result.append(elem)
            for elem in transTypes:
                if elem == TransitionType.COMPLETE:
                    result.append(elem)
            for elem in transTypes:
                if elem == TransitionType.SHIFT:
                    result.append(elem)
        else:
            for elem in transTypes:
                if elem == MWTTransitionType.MERGE_AS_MWT_VPC or elem == MWTTransitionType.MERGE_AS_MWT_LVC or \
                        elem == MWTTransitionType.MERGE_AS_MWT_ID or elem == MWTTransitionType.MERGE_AS_MWT_IREFLV or elem == MWTTransitionType.MERGE_AS_MWT_IAV or elem == MWTTransitionType.MERGE_AS_MWT_MVC:
                    result.append(elem)
            for elem in transTypes:
                if elem == TransitionType.MERGE_AS_ID or elem == TransitionType.MERGE_AS_IReflV or \
                        elem == TransitionType.MERGE_AS_VPC or elem == TransitionType.MERGE_AS_LVC or \
                        elem == TransitionType.MERGE_AS_OTH or elem == TransitionType.MERGE_AS_IAV or \
                        elem == TransitionType.MERGE_AS_MVC:
                    result.append(elem)
            for elem in transTypes:
                if elem == TransitionType.WHITE_MERGE:
                    result.append(elem)
            for elem in transTypes:
                if elem == TransitionType.REDUCE:
                    result.append(elem)
            for elem in transTypes:
                if elem == TransitionType.SHIFT:
                    result.append(elem)
        return result


class EmbedTransType(Enum):
    SHIFT = 0
    REDUCE = 4
    WHITE_MERGE = 5
    MERGE_AS_VPC_FULL = 6
    MERGE_AS_VPC_SEMI = 7
    MERGE_AS_IReflV = 8
    MERGE_AS_OTH = 9
    MERGE_AS_ID = 10
    MERGE_AS_LVC_FULL = 11
    MERGE_AS_LVC_CAUSE = 12
    MERGE_AS_MWT = 13
    MERGE_AS_MVC = 14
    MERGE_AS_IAV = 15
    MERGE_AS_LS_ICV = 16


class MWTTransitionType(Enum):
    MERGE_AS_MWT_VPC_FULL = 17
    MERGE_AS_MWT_VPC_SEMI = 18
    MERGE_AS_MWT_IREFLV = 19
    MERGE_AS_MWT_ID = 20
    MERGE_AS_MWT_LVC_FULL = 21
    MERGE_AS_MWT_LVC_CAUSE = 22
    MERGE_AS_MWT_OTH = 23
    MERGE_AS_MWT_MVC = 24
    MERGE_AS_MWT_IAV = 25
    MERGE_AS_MWT_LS_ICV = 26

