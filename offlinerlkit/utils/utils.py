from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

def d4rl_score(task, rew_mean, len_mean):
    score = ((rew_mean - REF_MIN_SCORE[task]) / (REF_MAX_SCORE[task] - REF_MIN_SCORE[task])) * 100

    return score
