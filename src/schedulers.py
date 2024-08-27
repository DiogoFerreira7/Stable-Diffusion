from diffusers import LMSDiscreteScheduler, SchedulerMixin, ConfigMixin

# Karras scheduler paper - https://arxiv.org/abs/2206.00364
# TODO perhaps trying to inherit LMSDiscreteScheduler and then overriding the methods used in the actual code could work
class CustomScheduler(SchedulerMixin):

    def __init__(self):
        super().__init__()