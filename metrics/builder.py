from common.register import Register
from .fetcher import MetricsFetcher

METRICS = Register(name="metrics")

def build_metrics_fetcher(metrics: str):
    if metrics == "" or metrics is None:
        return lambda pred, target, **kwargs: ""
    metric_list = metrics.split(",")
    metric_func = [METRICS.build(dict(module_name=m)) for m in metric_list]
    return MetricsFetcher(metric_list, metric_func)