import jittor as jt 
from jittor import nn 
from jdet.utils.registry import MODELS
from jittor import init, Module, Var

@MODELS.register_module()
def YoloParameterGroupsGenerator(model=None, weight_decay=-1, batch_size=64, accumulate=None, nominal_batch_size=64, named_params=None):
    nbs = nominal_batch_size  # YOLO nominal batch size
    if accumulate is None:
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    else:
        accumulate = max(int(accumulate), 1)
    # Match Ultralytics practice: scale weight decay by effective batch size.
    weight_decay *= batch_size * accumulate / nbs
    print(f"Scaled weight_decay = {weight_decay}")
    normal_group = {'params': []}
    weight_group = {'params': [], 'weight_decay': weight_decay} if weight_decay > 0 else {'params': []}
    bias_group = {'params': []} 
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, jt.Var):
            bias_group['params'].append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm):
            normal_group['params'].append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, jt.Var):
            weight_group['params'].append(v.weight)  # apply decay

    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(bias_group['params']), len(weight_group['params']), len(normal_group['params'])))
    return [normal_group, weight_group, bias_group]