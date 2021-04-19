from ray.rllib.agents.ppo.ppo_torch_policy import *

class CustomKLCoeffMixin(KLCoeffMixin):
    """ This is a custom version of the KLCoeffMixin to change the kl
        update rule to better line up with what was presented in the original paper.
    """

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        if sampled_kl > 1.5 * self.kl_target:
            self.kl_coeff *= 2.0
        elif sampled_kl < self.kl_target/1.5:
            self.kl_coeff /= 2.0
        # Return the current KL value.
        return self.kl_coeff

CustomPPOTorchPolicy = build_policy_class(
    name="PPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, CustomKLCoeffMixin,
        ValueNetworkMixin
    ],
)
