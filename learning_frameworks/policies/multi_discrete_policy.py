from typing import Tuple, List, Callable

import torch

from learning_frameworks.policies.policy import Policy


class MultiDiscretePolicy(Policy):
    @property
    def n_actions(self):
        return len(self.bins)

    def __init__(self, bins: List[int], input_size: int, output_size: int, layer_sizes: List[int],
                 activation_fn: Callable[[], torch.nn.Module], learning_rate: float, *args, **kwargs):
        super().__init__(input_size, output_size, layer_sizes, activation_fn, learning_rate, *args, **kwargs)
        self.bins = bins

    def __get_distrib_probs_from_output(self, outputs) -> torch.Tensor:
        """
        Get distribution probabilities from actor output

        Suppose:
            bins: [3, 3]

            n_procs: 5

        The output of the actor will be (5, sum(n_bins) *(6)* )

        We split into the logits into their respective bins to get
            (n_bins *(2)* , 5, 3)


        Since in the end, you want 2 actions for 5 processes and not 5 actions for 2 processes, you have to swap the first 2 dims to get
            (5, n_bins *(2)* , 3)

        Softmax everything on the last dim to get probabilities [0.4, 0.4, 0.2] for example

        Upon using these in categorical, you'll get
            (5, n_bins *(2)* )

        which is the number of actions you want (for 3 -> {0, 1, 2})


        :param outputs:
        :return:
        """

        # Split outputs into their corresponding bins
        # Result: list(n_bins, n_procs, n_actions)
        # TODO: if [3, 3, 2] this will break the stack
        splits = torch.split(outputs, self.bins, dim=-1)
        max_bin = max(self.bins)

        # Split returns a tuple, so we stack those to create another tensor
        # Result: Tensor(n_bins, n_procs, n_actions)
        tensor_splits = torch.zeros(len(self.bins), outputs.shape[0], max_bin)
        for i, split in enumerate(splits):
            padded_split = torch.nn.functional.pad(split, pad=(0, max_bin - self.bins[i], 0, 0), value=-torch.inf)
            tensor_splits[i] = padded_split
        splits = tensor_splits

        # Swap the splits so you get
        # Tensor(n_procs, n_bins, n_actions)
        splits = splits.swapdims(0, 1)

        # Softmax these to get
        # Tensor(n_procs, n_bins, n_actions)
        softm_splits = torch.softmax(splits, dim=-1)
        return softm_splits


    def act(self, input_, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fw_result = self.forward(input_)

        n_outputs = fw_result.shape[-1]

        distrib_probs = self.__get_distrib_probs_from_output(fw_result)

        if deterministic:
            return torch.argmax(distrib_probs, dim=-1), torch.zeros(size=(n_outputs, 1)), torch.zeros(size=(n_outputs, 1))
        else:
            distribution = torch.distributions.Categorical(distrib_probs)
            actions = distribution.sample()
            return actions, distribution.entropy().sum(-1), distribution.log_prob(actions).sum(-1)


    def get_backprop_data(self, observations, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        fw_result = self.forward(observations)
        distrib_probs = self.__get_distrib_probs_from_output(fw_result)

        distribution = torch.distributions.Categorical(distrib_probs)

        return distribution.entropy().sum(-1), distribution.log_prob(actions).sum(-1)