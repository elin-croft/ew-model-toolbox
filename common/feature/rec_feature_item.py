from collections import OrderedDict

class FeatureItem:
    def __init__(self,
        ordered_feature: OrderedDict = None,
        label: list = None
    ):
        self.feature = None
        self.ordered_feature = ordered_feature
        self.label = label