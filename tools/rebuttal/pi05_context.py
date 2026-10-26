"""Attach step context to an unmodified OpenPI PyTorch policy, without editing OpenPI."""
from functools import wraps
from gr00t.quantization.dit_step_context import set_dit_quant_step


def configure_policy(policy, steps):
    model = policy._model
    if not hasattr(model, 'denoise_step') or not hasattr(model, 'sample_actions'):
        raise RuntimeError('OpenPI model must expose sample_actions and denoise_step')
    if getattr(model, '_rebuttal_context_installed', False):
        if model._rebuttal_num_steps != steps:
            raise RuntimeError('Create a fresh policy for a different denoising schedule')
        return
    denoise = model.denoise_step
    sample = model.sample_actions
    # Strip torch.compile wrapping: per-step context and instrumentation require eager execution.
    while hasattr(sample, '_torchdynamo_orig_callable'):
        sample = sample._torchdynamo_orig_callable
    state = {'index': 0}

    @wraps(denoise)
    def denoise_with_context(*args, **kwargs):
        index = state['index']
        if index >= steps:
            raise RuntimeError(f'OpenPI made more than {steps} denoising calls')
        state['index'] += 1
        with set_dit_quant_step(index, total=steps):
            return denoise(*args, **kwargs)

    @wraps(sample)
    def sample_with_context(*args, **kwargs):
        state['index'] = 0
        kwargs['num_steps'] = steps
        with set_dit_quant_step(None):
            result = sample(*args, **kwargs)
        if state['index'] != steps:
            raise RuntimeError(f'Expected {steps} denoising calls, observed {state["index"]}')
        return result

    model.denoise_step = denoise_with_context
    model.sample_actions = sample_with_context
    policy._sample_actions = sample_with_context  # Policy caches the original bound method.
    policy._sample_kwargs = dict(getattr(policy, '_sample_kwargs', {}), num_steps=steps)
    model._rebuttal_context_installed = True
    model._rebuttal_num_steps = steps
