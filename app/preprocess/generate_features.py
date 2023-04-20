from preprocess.min_max import min_max

def generate_features(complexity: str, action: str, clarification: str) -> int:

    actions = ['add new', 'remove', 'troubleshoot', 'update']
    clarifications = ['content', 'feature', 'maintenance', 'other', 'page']
    combined = actions + clarifications
    if complexity > 5 or complexity < 1:
        raise ValueError('Invalid input for complexity. Must be between 1 - 5')
    if action not in actions:
        raise ValueError(
            f"Invalid input for action. Must be one of these options: {actions}")
    if clarification not in clarifications:
        raise ValueError(
            f"Invalid input for clarification. Must be one of these options: {clarification}")

    action_index = 0
    clarification_index = 0
    for i, val in enumerate(combined):
        if val == action:
            action_index = i + 1
        if val == clarification:
            clarification_index = i + 1

    result = np.zeros((1, len(combined) + 1), dtype=float)
    result[0, 0] = min_max(float(complexity))
    result[0, action_index] = 1
    result[0, clarification_index] = 1
