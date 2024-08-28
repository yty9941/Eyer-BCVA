import yaml

def config_loader():
    with open("default.yaml", 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    return dst_cfgs

def get_param_count(model):
    param_count_nogrd = 0
    param_count_grd = 0
    for param in model.parameters():
        if param.requires_grad:
            param_count_grd += param.size().numel()
        else:
            param_count_nogrd += param.size().numel()
    return param_count_grd, param_count_nogrd

def summary(mdoel, half = False):
    layers_count = len(list(mdoel.modules()))
    print(f"Model {mdoel} has {layers_count} layers.")
    param_grd, param_nogrd = get_param_count(mdoel)
    param_total = param_grd + param_nogrd
    print(f"-> Total number of parameters: {param_total:n}")
    print(f"-> Trainable parameters:       {param_grd:n}")
    print(f"-> Non-trainable parameters:   {param_nogrd:n}")
    approx_size = param_total * (2.0 if half else 4.0) * 10e-7
    print(f"Uncompressed size of the weights: {approx_size:.1f}MB")


