"""REINFORCE-Analysis plus frozen-reference KL, for every RL context."""
from .reinforce import analytic_probability_term_loss

def reference_kl(probabilities, log_probs, reference_log_probs, option_mask):
    import torch
    if probabilities.shape!=reference_log_probs.shape or option_mask.shape!=probabilities.shape:
        raise ValueError('reference and actor must cover the identical option set')
    ref=reference_log_probs.detach().float()
    if not torch.allclose(torch.logsumexp(ref.masked_fill(~option_mask,-torch.inf),-1),torch.zeros_like(ref[:,0]),atol=2e-3,rtol=0):
        raise ValueError('reference distribution must be normalized')
    delta=torch.where(option_mask,log_probs.float()-ref,torch.zeros_like(ref))
    return (probabilities.float()*delta).sum(-1).mean()

def reinforce_a_primary(probabilities, log_probs, actions, feedback, behavior_action_probs,
                        reference_log_probs, option_mask, *, kl_coefficient=.01):
    result=analytic_probability_term_loss(probabilities,log_probs,actions,feedback,behavior_action_probs,
        alpha=.5,option_mask=option_mask,baseline='loo')
    return result.loss + kl_coefficient*reference_kl(probabilities,log_probs,reference_log_probs,option_mask),result
