from typing import List, Callable
# from .builder import METRICS

class MetricsFetcher:
    def __init__(self, metrics:List[str], metrics_func:List[Callable]):
        self.metrics = metrics
        for i, t in enumerate(metrics):
            self.metircs_map[t] = metrics_func[i]
    
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