def get_action_params(action_space, action_dist_params):
    if action_space["cls"] == "discrete":
        num_actions = action_space["high"]
        max_action = num_actions
        if action_dist_params is None:
            action_dist_params = dict(cls="Categorical")  # same as `torch.multinomial`

    elif action_space["cls"] == "box":
        num_actions = action_space["shape"][0]
        try:
            max_action = max(action_space["high"])
        except:  # TypeError: 'float' object is not iterable
            max_action = action_space["high"]
        if max_action != 1.0:
            raise RuntimeError("Currently only supporting continuous action spaces bound to [-1.0, 1.0]")

        if action_dist_params is None:
            action_dist_params = dict(cls="SquashedDiagGaussian")
    else:
        raise ValueError

    return num_actions, max_action, action_dist_params
