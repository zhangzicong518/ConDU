import torch


class DeltaModel():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, delta_model=None):
        """Initializes the delta model from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the delta model state dict.
        """
        if delta_model is not None:
            self.delta_model = delta_model
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = pretrained_checkpoint.state_dict()
                finetuned_state_dict = finetuned_checkpoint.state_dict()
                self.delta_model = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.delta_model[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two delta models together."""
        with torch.no_grad():
            new_delta_model = {}
            for key in self.delta_model:
                if key not in other.delta_model:
                    print(f'Warning, key {key} is not present in both delta models.')
                    continue
                new_delta_model[key] = self.delta_model[key] + other.delta_model[key]
        return DeltaModel(delta_model=new_delta_model)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a delta model."""
        with torch.no_grad():
            new_delta_model = {}
            for key in self.delta_model:
                new_delta_model[key] = - self.delta_model[key]
        return DeltaModel(delta_model=new_delta_model)

    def weightmerging(self, delta_models, coefficients):
        with torch.no_grad():
            new_delta_model = {}
            for key in delta_models[0].delta_model:
                new_delta_model[key] = sum(coefficients[k] * delta_models[k][key] for k in range(len(delta_models)))
        return DeltaModel(delta_model=new_delta_model)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a delta model to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_checkpoint.state_dict()
            for key in pretrained_state_dict:
                if key not in self.delta_model:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the delta model')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.delta_model[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

def apply_delta_model(delta_model, pretrained_model):
    """Apply a delta model to a pretrained model."""
    with torch.no_grad(): 
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in delta_model:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the delta model')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + delta_model[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model


def unify_delta_models(delta_models, device=None):
    target_device = None
    if device is None:
        target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        target_device = device

    sum_param = {}
    n2p = []
    for m in range(len(delta_models)):
        n2p_temp = delta_models[m].delta_model
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n].to(target_device))
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    delta_model_unified = {}
    scales = torch.zeros(len(delta_models), device=target_device)
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n]).to(target_device)
        for m in range(len(delta_models)):
            param = delta_models[m].delta_model[n]
            param = param.to(target_device)
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        delta_model_unified[n] = param_max * flag
    new_scales = torch.zeros(len(delta_models), device=target_device)
    for m in range(len(delta_models)):
        for n in delta_model_unified:
            p = delta_model_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return delta_model_unified, masks, rescalers
