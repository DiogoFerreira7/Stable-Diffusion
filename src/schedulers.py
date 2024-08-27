# TODO To implement the schedulers try use the SchedulerMixin that will implement all the APIs

# TODO go over lesson 22 again to implement the schedulers and understand the maths behind all of it
from diffusers import SchedulerMixin, ConfigMixin

class CustomScheduler(SchedulerMixin):

    def __init__(self):
        super().__init__()