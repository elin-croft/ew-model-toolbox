# from .models import BACKBONE, RECMODEL
# from .losses import LOSS
from .register import Register

RECMODEL=Register("rec_models")
BACKBONE=Register("backbone")
LOSS=Register("loss")
