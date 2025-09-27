from typing import AnyStr
from .builder import METRICS
class MetricsFetcher:
    def __init__(self, metrics:str):
        self.metrics = metrics.split(",")
        self.metircs_map = {k: METRICS.build((dict(module_name=k))) for k in self.metrics}
    
    def __call__(self, pred, target, **kwargs):
        result_list = []
        for k in self.metircs:
            metric = self.metircs_map[k](pred, target, **kwargs)
            result_list.append(f'{k}: {metric}')
        return ", ".join(result_list)
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.metrics:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string